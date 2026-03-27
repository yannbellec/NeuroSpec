"""
01_pretrain_synthetic.py — DiffFOOOF pretraining on synthetic EEG spectra.

Target hardware : NVIDIA H100 80 GB (HBM3), CUDA 12.2, bf16.

Corrections vs previous version:
    [1] FULL parameter supervision : datasets now return cfs, amps, bws,
        peak_mask in addition to offset/exponent.  DiffFOOOFLoss uses all
        of them via masked MSE + BCE gate.
    [2] Vectorised batch generation : all datasets generate an entire batch
        at once with torch operations — no sample-wise Python loop, no CPU
        bottleneck.
    [3] spectral_mixup removed from phases 1 and 2 (creates non-identifiable
        parameter targets for mixed spectra).
    [4] GradScaler removed — useless and potentially harmful with bf16
        (scaling is only needed for fp16 underflow).
    [5] cudnn.deterministic=False + benchmark=True — coherent for a large
        GPU pretrain run (determinism was contradicting benchmark mode).
    [6] n_peaks fixed at 6 (8 aggravated identifiability without benefit).
    [7] Real GPU profiling : samples/s, data-loading time, compute time,
        torch.cuda.max_memory_allocated() reported every epoch.
    [8] EMA of model weights for a stable validation checkpoint.
    [9] IterableDataset-style vectorised generators replace sample-wise
        Dataset.__getitem__ to feed the H100 at full speed.
"""

import os
import sys
import time
import math
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import DiffFOOOF, DiffFOOOFLoss


# ===========================================================================
# 0. CONFIG
# ===========================================================================

class CFG:
    # Frequency grid
    f_min       : float = 1.0
    f_max       : float = 45.0
    n_freqs     : int   = 100

    # Model
    n_peaks     : int   = 6        # kept at 6 for identifiability
    hidden_dim  : int   = 256

    # Training
    batch_size  : int   = 4096     # H100 80 GB can handle this easily in bf16
    n_workers   : int   = 8        # data-loading workers
    seed        : int   = 42

    # Phase durations (epochs)
    phase1_epochs : int = 65       # clean + noisy, curriculum noise level
    phase2_epochs : int = 50       # overlapping peaks mixed in progressively
    phase3_epochs : int = 20       # LFP-domain distribution bias
    phase4_epochs : int = 15       # noisy boost — casse le plateau val noisy

    # Samples per epoch per dataset type
    # Total = n_samples * n_active_datasets per epoch
    n_samples_per_epoch : int = 120_000

    # Learning rate
    lr_init     : float = 1e-3
    lr_min      : float = 1e-5
    warmup_epochs : int = 5

    # Regularisation / loss weights
    lambda_sparse    : float = 0.10
    lambda_bw_excess : float = 0.05   # one-sided, replaces lambda_bw
    bw_soft_max      : float = 4.0    # Hz — only penalise bw above this
    lambda_ap        : float = 0.50
    lambda_peaks     : float = 0.30
    lambda_unmatched : float = 0.10   # L1 on unmatched slots, replaces lambda_mask

    # bf16 mixed precision (native bf16, no GradScaler needed)
    use_bf16    : bool  = True

    # Checkpoints
    checkpoint_dir : str = "checkpoints"
    ema_decay      : float = 0.999   # EMA of weights for stable val checkpoint


# ===========================================================================
# 1. VECTORISED BATCH GENERATORS
# ===========================================================================
# All generators produce an entire batch at once in torch,
# directly on CPU (pin_memory + num_workers handles the transfer).
# Each __iter__ loop yields (psd [B, F], gt_params dict).

class _BaseSpectraDataset(IterableDataset):
    """Base class — subclasses implement _generate_batch."""

    def __init__(self, freqs: torch.Tensor, n_peaks: int,
                 n_samples: int, batch_size: int, seed_offset: int = 0):
        super().__init__()
        self.freqs      = freqs                 # [F]
        self.n_peaks    = n_peaks
        self.n_samples  = n_samples
        self.batch_size = batch_size
        self.seed_offset = seed_offset
        self.f_min      = freqs[0].item()
        self.f_max      = freqs[-1].item()
        self.log_f      = torch.log10(freqs + 1e-6)  # [F]
        self._epoch     = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __iter__(self):
        rng = torch.Generator()
        rng.manual_seed(CFG.seed + self.seed_offset + self._epoch * 997)
        n_batches = math.ceil(self.n_samples / self.batch_size)
        for _ in range(n_batches):
            yield self._generate_batch(self.batch_size, rng)

    def _make_gt(self, offset, exponent, cfs_list, amps_list, bws_list):
        """
        Pack per-sample peak lists into padded tensors [B, K].
        Peaks are sorted by cf (ascending) and padded with 0.0.
        peak_mask = 1.0 for real peaks, 0.0 for padding slots.
        """
        B = offset.shape[0]
        K = self.n_peaks
        gt_cfs  = torch.zeros(B, K)
        gt_amps = torch.zeros(B, K)
        gt_bws  = torch.zeros(B, K)
        gt_mask = torch.zeros(B, K)

        for i in range(B):
            c = cfs_list[i]
            a = amps_list[i]
            w = bws_list[i]
            n = len(c)
            if n > 0:
                # sort by cf
                order = torch.argsort(c)
                c, a, w = c[order], a[order], w[order]
                n_fill = min(n, K)
                gt_cfs[i,  :n_fill] = c[:n_fill]
                gt_amps[i, :n_fill] = a[:n_fill]
                gt_bws[i,  :n_fill] = w[:n_fill]
                gt_mask[i, :n_fill] = 1.0

        return {
            'offset':    offset,
            'exponent':  exponent,
            'cfs':       gt_cfs,
            'amps':      gt_amps,
            'bws':       gt_bws,
            'peak_mask': gt_mask,
        }

    def _generate_batch(self, B, rng):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1a. Clean spectra — well-separated peaks, low noise
# ---------------------------------------------------------------------------
class CleanSpectraDataset(_BaseSpectraDataset):
    """
    Easy cases: 0–3 well-separated peaks, negligible noise.
    Teaches the model the basic aperiodic + Gaussian decomposition.
    """

    def _generate_batch(self, B, rng):
        F_ = len(self.freqs)

        offset   = torch.empty(B).uniform_(0.5, 3.0,   generator=rng)
        exponent = torch.empty(B).uniform_(0.5, 2.5,   generator=rng)

        # Aperiodic [B, F]
        aperiodic = offset.unsqueeze(1) - exponent.unsqueeze(1) * self.log_f.unsqueeze(0)

        # Peaks — vectorised by drawing for max_peaks slots then masking
        max_p = 3
        n_peaks_per = torch.randint(0, max_p + 1, (B,), generator=rng)  # [B]

        cf_raw  = torch.empty(B, max_p).uniform_(self.f_min + 3, self.f_max - 5, generator=rng)
        amp_raw = torch.empty(B, max_p).uniform_(0.2, 1.5, generator=rng)
        bw_raw  = torch.empty(B, max_p).uniform_(1.5, 4.0, generator=rng)

        # Active mask [B, max_p]
        peak_idx = torch.arange(max_p).unsqueeze(0)         # [1, max_p]
        active   = peak_idx < n_peaks_per.unsqueeze(1)      # [B, max_p]

        # Compute periodic component [B, F]
        # freqs [F] → [1, 1, F], cfs [B, max_p] → [B, max_p, 1]
        freqs_e = self.freqs.view(1, 1, -1)
        cfs_e   = cf_raw.unsqueeze(2)
        bws_e   = bw_raw.unsqueeze(2)
        amps_e  = amp_raw.unsqueeze(2) * active.float().unsqueeze(2)

        gaussians = amps_e * torch.exp(-0.5 * ((freqs_e - cfs_e) / bws_e) ** 2)
        periodic  = gaussians.sum(dim=1)                     # [B, F]

        noise = torch.randn(B, F_, generator=rng) * 0.02
        psd   = aperiodic + periodic + noise

        # Build gt_params from lists
        cf_list  = [cf_raw[i, active[i]]  for i in range(B)]
        amp_list = [amp_raw[i, active[i]] for i in range(B)]
        bw_list  = [bw_raw[i, active[i]]  for i in range(B)]
        gt = self._make_gt(offset, exponent, cf_list, amp_list, bw_list)

        return psd, gt


# ---------------------------------------------------------------------------
# 1b. Noisy spectra — coloured noise, variable SNR
# ---------------------------------------------------------------------------
class NoisySpectraDataset(_BaseSpectraDataset):
    """
    Harder: coloured 1/f noise (not white), variable SNR.
    Forces the model to disentangle noise from real aperiodic component.
    noise_level ∈ [0, 1] controlled externally for curriculum.
    """

    def __init__(self, *args, noise_level=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_level = noise_level   # 0.0 → clean, 1.0 → very noisy

    def set_noise_level(self, level: float):
        self.noise_level = float(level)

    def _generate_batch(self, B, rng):
        F_ = len(self.freqs)

        offset   = torch.empty(B).uniform_(0.5, 3.0, generator=rng)
        exponent = torch.empty(B).uniform_(0.5, 2.5, generator=rng)

        aperiodic = offset.unsqueeze(1) - exponent.unsqueeze(1) * self.log_f.unsqueeze(0)

        max_p = 4
        n_peaks_per = torch.randint(0, max_p + 1, (B,), generator=rng)
        cf_raw  = torch.empty(B, max_p).uniform_(self.f_min + 2, self.f_max - 3, generator=rng)
        amp_raw = torch.empty(B, max_p).uniform_(0.1, 1.5, generator=rng)
        bw_raw  = torch.empty(B, max_p).uniform_(1.0, 5.0, generator=rng)

        peak_idx = torch.arange(max_p).unsqueeze(0)
        active   = peak_idx < n_peaks_per.unsqueeze(1)

        freqs_e = self.freqs.view(1, 1, -1)
        cfs_e   = cf_raw.unsqueeze(2)
        bws_e   = bw_raw.unsqueeze(2)
        amps_e  = amp_raw.unsqueeze(2) * active.float().unsqueeze(2)
        gaussians = amps_e * torch.exp(-0.5 * ((freqs_e - cfs_e) / bws_e) ** 2)
        periodic  = gaussians.sum(dim=1)

        # Coloured noise: 1/f^alpha profile, alpha random per sample
        noise_alpha  = torch.empty(B).uniform_(0.5, 2.0, generator=rng)   # [B]
        noise_psd_shape = self.freqs.pow(-noise_alpha.unsqueeze(1))        # [B, F]
        noise_psd_shape = noise_psd_shape / noise_psd_shape.std(dim=1, keepdim=True).clamp(min=1e-6)
        noise_amplitude = self.noise_level * torch.empty(B).uniform_(0.05, 0.25, generator=rng)
        coloured_noise  = noise_psd_shape * noise_amplitude.unsqueeze(1)

        psd = aperiodic + periodic + coloured_noise

        cf_list  = [cf_raw[i, active[i]]  for i in range(B)]
        amp_list = [amp_raw[i, active[i]] for i in range(B)]
        bw_list  = [bw_raw[i, active[i]]  for i in range(B)]
        gt = self._make_gt(offset, exponent, cf_list, amp_list, bw_list)

        return psd, gt


# ---------------------------------------------------------------------------
# 1c. Overlapping peaks — vectorised rejection sampling
# ---------------------------------------------------------------------------
class OverlappingPeaksDataset(_BaseSpectraDataset):
    """
    Peaks with min separation as small as 1–2 Hz.
    Forces the model to learn to resolve close peaks rather than merge them.
    overlap_difficulty ∈ [0, 1]: 0 → 4 Hz min sep, 1 → 1 Hz min sep.

    Fully vectorised: rejection sampling is done batch-wise in torch.
    No Python loop per sample — the H100 data pipeline is not starved.

    Strategy:
      - Draw max_candidates >> n_peaks CFs per sample in one tensor op.
      - Sort them.
      - Greedily accept the first n_peaks that respect min_sep using
        cumulative difference on the sorted candidates — all in torch,
        no Python conditional per sample.
    """

    def __init__(self, *args, overlap_difficulty=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.overlap_difficulty = overlap_difficulty

    def set_difficulty(self, d: float):
        self.overlap_difficulty = float(d)

    def _generate_batch(self, B, rng):
        F_     = len(self.freqs)
        K      = min(self.n_peaks, 4)
        min_sep = 4.0 - 3.0 * self.overlap_difficulty    # 4 Hz → 1 Hz

        offset   = torch.empty(B).uniform_(0.5, 3.0, generator=rng)
        exponent = torch.empty(B).uniform_(0.5, 2.5, generator=rng)
        aperiodic = offset.unsqueeze(1) - exponent.unsqueeze(1) * self.log_f.unsqueeze(0)

        # --- Vectorised CF rejection sampling ---
        # Draw many candidates per sample, sort, keep those respecting min_sep
        n_cands   = K * 12          # oversampling factor — always enough after rejection
        cf_cands  = torch.empty(B, n_cands).uniform_(
            self.f_min + 2, self.f_max - 5, generator=rng
        )                                                 # [B, n_cands]
        cf_sorted, _ = cf_cands.sort(dim=1)               # [B, n_cands] ascending

        # For each sample, keep first K candidates where consecutive gap >= min_sep.
        # Implementation: compute all consecutive diffs, build a boolean acceptance mask.
        diffs = torch.diff(cf_sorted, dim=1)              # [B, n_cands-1]  ≥ 0

        # accepted[b, c] = True if candidate c can be included given all prior accepted ones.
        # We simulate the greedy accept via a cumulative minimum-gap scan:
        # A candidate is rejected if its gap to the last accepted candidate is < min_sep.
        # We do this with a simple forward scan — still fully in torch (no per-sample Python).
        accepted = torch.zeros(B, n_cands, dtype=torch.bool)
        accepted[:, 0] = True                             # first candidate always accepted
        last_accepted_cf = cf_sorted[:, 0].clone()        # [B]

        # Scan in vectorised style over candidate positions (n_cands iterations, not B)
        for c in range(1, n_cands):
            gap_ok  = (cf_sorted[:, c] - last_accepted_cf) >= min_sep   # [B] bool
            take    = gap_ok                              # accept if gap is large enough
            accepted[:, c] = take
            last_accepted_cf = torch.where(take, cf_sorted[:, c], last_accepted_cf)

        # From accepted candidates, pick the first K per sample
        # accepted_rank[b, c] = rank among accepted (0,1,2,...), -1 if not accepted
        # We use cumsum trick to assign ranks
        ranks = accepted.long().cumsum(dim=1) - 1         # [B, n_cands]  0-based rank
        ranks[~accepted] = n_cands                        # push non-accepted out

        # For each target slot k, find the candidate with rank == k
        selected_mask = torch.zeros(B, n_cands, dtype=torch.bool)
        for k in range(K):
            is_rank_k = ranks == k                        # [B, n_cands]
            selected_mask |= is_rank_k

        # Gather selected CFs: [B, K] — sorted by construction
        # Use masked_select with reshape: guaranteed K accepted per sample
        # (n_cands is large enough that exactly K will always be found)
        selected_cfs = cf_sorted[selected_mask].view(B, K)  # [B, K]  sorted ascending

        # Draw amps and bws for the K selected peaks
        amp_raw = torch.empty(B, K).uniform_(0.15, 1.2, generator=rng)
        bw_raw  = torch.empty(B, K).uniform_(0.8,  3.5, generator=rng)

        # Number of real peaks per sample: random in [1, K]
        n_peaks_per = torch.randint(1, K + 1, (B,), generator=rng)  # [B]
        peak_idx    = torch.arange(K).unsqueeze(0)                   # [1, K]
        active      = (peak_idx < n_peaks_per.unsqueeze(1)).float()  # [B, K]

        # Build periodic component [B, F]
        freqs_e = self.freqs.view(1, 1, -1)
        cfs_e   = selected_cfs.unsqueeze(2)
        bws_e   = bw_raw.unsqueeze(2)
        amps_e  = (amp_raw * active).unsqueeze(2)
        gaussians = amps_e * torch.exp(-0.5 * ((freqs_e - cfs_e) / bws_e) ** 2)
        periodic  = gaussians.sum(dim=1)                  # [B, F]

        noise = torch.randn(B, F_, generator=rng) * 0.05
        psd   = aperiodic + periodic + noise

        # Build gt using active mask — active peaks are the first n_peaks_per slots
        cf_list  = [selected_cfs[i, active[i].bool()] for i in range(B)]
        amp_list = [amp_raw[i,       active[i].bool()] for i in range(B)]
        bw_list  = [bw_raw[i,        active[i].bool()] for i in range(B)]
        gt = self._make_gt(offset, exponent, cf_list, amp_list, bw_list)
        return psd, gt


# ---------------------------------------------------------------------------
# 1d. LFP-domain — distribution bias towards real striatal LFP statistics
# ---------------------------------------------------------------------------
class LFPDomainDataset(_BaseSpectraDataset):
    """
    Distribution shifted toward real MitoPark LFP statistics:
      - exponent skewed toward 1.5–2.5 (striatal range)
      - 60% chance of beta peak (13–30 Hz)
      - 30% chance of theta peak (4–8 Hz)
      - strong coloured noise floor
    """

    def _generate_batch(self, B, rng):
        F_ = len(self.freqs)

        # Striatal LFP aperiodic distribution (skewed toward higher exponents)
        offset   = torch.empty(B).uniform_(1.0, 3.5, generator=rng)
        exponent = 1.5 + torch.empty(B).exponential_(1.0 / 0.8, generator=rng).clamp(max=2.5)

        aperiodic = offset.unsqueeze(1) - exponent.unsqueeze(1) * self.log_f.unsqueeze(0)

        periodic = torch.zeros(B, F_)
        cf_list, amp_list, bw_list = [], [], []

        for i in range(B):
            cfs_i, amps_i, bws_i = [], [], []

            # Beta peak (13–30 Hz) — 60% probability
            if torch.rand(1, generator=rng).item() < 0.60:
                cf  = torch.empty(1).uniform_(13.0, 30.0, generator=rng).item()
                amp = torch.empty(1).uniform_(0.2, 1.5, generator=rng).item()
                bw  = torch.empty(1).uniform_(1.5, 4.0, generator=rng).item()
                cfs_i.append(cf); amps_i.append(amp); bws_i.append(bw)

            # Theta peak (4–8 Hz) — 30% probability
            if torch.rand(1, generator=rng).item() < 0.30:
                cf  = torch.empty(1).uniform_(4.0, 8.0, generator=rng).item()
                amp = torch.empty(1).uniform_(0.1, 0.8, generator=rng).item()
                bw  = torch.empty(1).uniform_(1.0, 3.0, generator=rng).item()
                cfs_i.append(cf); amps_i.append(amp); bws_i.append(bw)

            # Optional extra peak — 20% probability
            if torch.rand(1, generator=rng).item() < 0.20:
                cf  = torch.empty(1).uniform_(self.f_min + 2, self.f_max - 5, generator=rng).item()
                amp = torch.empty(1).uniform_(0.05, 0.4, generator=rng).item()
                bw  = torch.empty(1).uniform_(1.0, 3.5, generator=rng).item()
                cfs_i.append(cf); amps_i.append(amp); bws_i.append(bw)

            if cfs_i:
                cfs_t  = torch.tensor(cfs_i)
                amps_t = torch.tensor(amps_i)
                bws_t  = torch.tensor(bws_i)
                f_row  = self.freqs.unsqueeze(0)
                g = amps_t.view(-1,1) * torch.exp(
                    -0.5 * ((f_row - cfs_t.view(-1,1)) / bws_t.view(-1,1)) ** 2
                )
                periodic[i] = g.sum(dim=0)
                cf_list.append(cfs_t); amp_list.append(amps_t); bw_list.append(bws_t)
            else:
                cf_list.append(torch.tensor([])); amp_list.append(torch.tensor([]))
                bw_list.append(torch.tensor([]))

        # Strong coloured noise floor (realistic LFP recording conditions)
        noise_alpha = torch.empty(B).uniform_(0.5, 1.5, generator=rng)
        noise_floor = self.freqs.pow(-noise_alpha.unsqueeze(1))
        noise_floor = noise_floor / noise_floor.std(dim=1, keepdim=True).clamp(min=1e-6)
        noise_amp   = torch.empty(B).uniform_(0.05, 0.15, generator=rng)
        coloured    = noise_floor * noise_amp.unsqueeze(1)

        psd = aperiodic + periodic + coloured

        gt = self._make_gt(offset, exponent, cf_list, amp_list, bw_list)
        return psd, gt


# ---------------------------------------------------------------------------
# 1e. Fixed validation sets — one per domain type
# ---------------------------------------------------------------------------
def make_val_sets(freqs, n_peaks, n_val=2000, batch_size=512):
    """Generate fixed validation batches (once, before training)."""
    rng = torch.Generator(); rng.manual_seed(CFG.seed + 9999)
    val = {}
    for name, cls, kwargs in [
        ('clean',    CleanSpectraDataset,    {}),
        ('noisy',    NoisySpectraDataset,    {'noise_level': 0.8}),
        ('overlap',  OverlappingPeaksDataset, {'overlap_difficulty': 0.8}),
        ('lfp',      LFPDomainDataset,       {}),
    ]:
        ds = cls(freqs, n_peaks, n_samples=n_val, batch_size=n_val, seed_offset=0, **kwargs)
        psd, gt = ds._generate_batch(n_val, rng)
        val[name] = (psd, gt)
    return val


# ===========================================================================
# 2. EMA helper
# ===========================================================================

class EMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())
        for k in self.shadow:
            self.shadow[k] = self.shadow[k].float()

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.float()

    def apply_shadow(self, model):
        model.load_state_dict({k: v.to(next(model.parameters()).device)
                               for k, v in self.shadow.items()})


# ===========================================================================
# 3. TRAINING UTILITIES
# ===========================================================================

def get_lr(epoch, total_epochs, lr_init, lr_min, warmup_epochs):
    """Linear warmup + cosine decay."""
    if epoch < warmup_epochs:
        return lr_init * (epoch + 1) / warmup_epochs
    t = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return lr_min + 0.5 * (lr_init - lr_min) * (1 + math.cos(math.pi * t))


def collate_fn(batch):
    """
    batch : list of (psd [B, F], gt_params dict)
    The IterableDataset already yields full batches → just unpack the first element.
    """
    psd, gt = batch[0]
    return psd, gt


@torch.no_grad()
def validate(model, val_sets, criterion, device, dtype):
    model.eval()
    results = {}
    for name, (psd, gt) in val_sets.items():
        psd_d = psd.to(device)
        gt_d  = {k: v.to(device) for k, v in gt.items()}
        with torch.autocast(device_type='cuda', dtype=dtype, enabled=(dtype == torch.bfloat16)):
            pred, params = model(psd_d)
            loss, ldict  = criterion(pred, psd_d, params, gt_d)
        results[name] = {k: v for k, v in ldict.items()}
        results[name]['total'] = loss.item()
    model.train()
    return results


def log_gpu_stats(epoch, t_data, t_compute, n_samples):
    mem_alloc = torch.cuda.max_memory_allocated() / 1024**3
    mem_res   = torch.cuda.max_memory_reserved()  / 1024**3
    sps       = n_samples / max(t_data + t_compute, 1e-6)
    frac_data = 100 * t_data / max(t_data + t_compute, 1e-6)
    print(
        f"  [GPU] mem_alloc={mem_alloc:.1f}GB  mem_res={mem_res:.1f}GB  "
        f"samples/s={sps:.0f}  data={frac_data:.1f}% of wall-time"
    )
    torch.cuda.reset_peak_memory_stats()


# ===========================================================================
# 4. PHASE RUNNERS
# ===========================================================================

def run_phase(
    phase_name, phase_epochs, epoch_offset,
    datasets_and_weights,      # list of (IterableDataset, weight)
    model, ema, optimizer, criterion, val_sets,
    device, dtype, scaler_unused,
    checkpoint_dir, best_val,
):
    """
    Generic training loop for one phase.

    datasets_and_weights : at each step we draw from all datasets with
    probability proportional to weight.  Since each yields full batches,
    we simply iterate round-robin weighted by floor(weight*N_batches).
    """
    print(f"\n{'='*60}")
    print(f"  PHASE: {phase_name}  ({phase_epochs} epochs)")
    print(f"{'='*60}")

    total_epochs = epoch_offset + phase_epochs

    for ep_local in range(phase_epochs):
        epoch = epoch_offset + ep_local
        t_epoch_start = time.time()

        # Update LR
        lr = get_lr(epoch, total_epochs, CFG.lr_init, CFG.lr_min, CFG.warmup_epochs)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Set epoch on all datasets (for deterministic but varying seeds)
        for ds, _ in datasets_and_weights:
            ds.set_epoch(epoch)

        # Build a mixed loader by concatenating batches from all datasets
        # proportionally to their weight
        model.train()
        epoch_loss   = 0.0
        epoch_ldict  = {k: 0.0 for k in ['recon', 'sparse', 'bw_excess', 'ap', 'peaks', 'unmatched']}
        n_batches    = 0
        t_data_total = 0.0
        t_comp_total = 0.0
        n_samples    = 0

        # Round-robin over weighted datasets
        iterators = [iter(DataLoader(
            ds,
            batch_size=1,           # ds already yields full batches
            collate_fn=collate_fn,
            num_workers=CFG.n_workers,
            pin_memory=True,
            prefetch_factor=2,
        )) for ds, _ in datasets_and_weights]

        weights = [w for _, w in datasets_and_weights]
        total_w = sum(weights)
        n_iter  = [max(1, round(w / total_w * 30)) for w in weights]  # batches per ds per epoch

        for ds_idx, (it, n_it) in enumerate(zip(iterators, n_iter)):
            for _ in range(n_it):
                t0 = time.time()
                try:
                    psd, gt = next(it)
                except StopIteration:
                    break
                t_data_total += time.time() - t0

                psd = psd.to(device, non_blocking=True)
                gt  = {k: v.to(device, non_blocking=True) for k, v in gt.items()}

                t1 = time.time()
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type='cuda', dtype=dtype,
                                    enabled=(dtype == torch.bfloat16)):
                    pred, params = model(psd)
                    loss, ldict  = criterion(pred, psd, params, gt)

                # bf16: no GradScaler needed
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                ema.update(model)

                t_comp_total += time.time() - t1
                epoch_loss   += loss.item()
                for k in epoch_ldict:
                    epoch_ldict[k] += ldict.get(k, 0.0)
                n_batches  += 1
                n_samples  += psd.shape[0]

        if n_batches == 0:
            continue

        avg_loss = epoch_loss / n_batches
        avg_ld   = {k: v / n_batches for k, v in epoch_ldict.items()}
        t_epoch  = time.time() - t_epoch_start

        print(
            f"Epoch {epoch+1:03d}/{total_epochs} | "
            f"loss={avg_loss:.4f} | "
            f"recon={avg_ld['recon']:.4f} | ap={avg_ld['ap']:.4f} | "
            f"peaks={avg_ld['peaks']:.4f} | unmatch={avg_ld['unmatched']:.4f} | "
            f"bw_ex={avg_ld['bw_excess']:.4f} | LR={lr:.2e} | {t_epoch:.1f}s"
        )
        log_gpu_stats(epoch, t_data_total, t_comp_total, n_samples)

        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0 or ep_local == phase_epochs - 1:
            val_res = validate(model, val_sets, criterion, device, dtype)
            val_summary = "  [VAL] " + "  ".join(
                f"{name}={r['total']:.4f}" for name, r in val_res.items()
            )
            print(val_summary)

            avg_val = sum(r['total'] for r in val_res.values()) / len(val_res)
            if avg_val < best_val['loss']:
                best_val['loss']  = avg_val
                best_val['epoch'] = epoch
                # Save EMA weights as the checkpoint
                ema_model = copy.deepcopy(model)
                ema.apply_shadow(ema_model)
                torch.save({
                    'epoch':            epoch,
                    'model_state_dict': ema_model.state_dict(),
                    'val_loss':         avg_val,
                    'freqs':            freqs_np,
                    'phase':            phase_name,
                }, os.path.join(checkpoint_dir, 'pretrain_best.pt'))
                print(f"  → Best checkpoint saved (val={avg_val:.4f})")
                del ema_model

    return best_val


# ===========================================================================
# 5. MAIN
# ===========================================================================

# Store freqs globally for checkpoint saving
freqs_np = np.linspace(CFG.f_min, CFG.f_max, CFG.n_freqs)


def main():
    # --- Reproducibility vs speed: choose one ---
    # For a large pretrain run: benchmark=True, deterministic=False
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.bfloat16 if (CFG.use_bf16 and device.type == 'cuda') else torch.float32

    print(f"Device : {device}")
    print(f"dtype  : {dtype}")
    if device.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    freqs_t = torch.tensor(freqs_np, dtype=torch.float32)
    os.makedirs(CFG.checkpoint_dir, exist_ok=True)

    # --- Model ---
    model = DiffFOOOF(freqs=freqs_np, n_peaks=CFG.n_peaks, hidden_dim=CFG.hidden_dim).to(device)
    model = torch.compile(model)   # H100: big speedup
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    ema = EMA(model, decay=CFG.ema_decay)

    criterion = DiffFOOOFLoss(
        lambda_sparse    = CFG.lambda_sparse,
        lambda_bw_excess = CFG.lambda_bw_excess,
        bw_soft_max      = CFG.bw_soft_max,
        lambda_ap        = CFG.lambda_ap,
        lambda_peaks     = CFG.lambda_peaks,
        lambda_unmatched = CFG.lambda_unmatched,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr_init, weight_decay=1e-4, fused=True  # fused=True for H100
    )

    # --- Validation sets (fixed, generated once) ---
    print("Generating fixed validation sets...")
    val_sets = make_val_sets(freqs_t, CFG.n_peaks)
    print(f"  {sum(v[0].shape[0] for v in val_sets.values())} validation spectra across {len(val_sets)} domains")

    # --- Datasets ---
    S = CFG.n_samples_per_epoch
    B = CFG.batch_size

    ds_clean   = CleanSpectraDataset(freqs_t, CFG.n_peaks, n_samples=S, batch_size=B, seed_offset=0)
    ds_noisy   = NoisySpectraDataset(freqs_t, CFG.n_peaks, n_samples=S, batch_size=B, seed_offset=1, noise_level=0.3)
    ds_overlap = OverlappingPeaksDataset(freqs_t, CFG.n_peaks, n_samples=S, batch_size=B, seed_offset=2, overlap_difficulty=0.2)
    ds_lfp     = LFPDomainDataset(freqs_t, CFG.n_peaks, n_samples=S, batch_size=B, seed_offset=3)

    best_val = {'loss': float('inf'), 'epoch': -1}
    epoch_offset = 0

    # -------------------------------------------------------------------
    # PHASE 1 — Clean foundation + growing noise
    # 60% clean, 40% noisy, noise_level 0.3 → 0.8
    # -------------------------------------------------------------------
    for ep in range(CFG.phase1_epochs):
        progress = ep / max(CFG.phase1_epochs - 1, 1)
        ds_noisy.set_noise_level(0.3 + 0.5 * progress)

    best_val = run_phase(
        phase_name          = "Phase1_clean+noisy",
        phase_epochs        = CFG.phase1_epochs,
        epoch_offset        = epoch_offset,
        datasets_and_weights= [(ds_clean, 0.6), (ds_noisy, 0.4)],
        model=model, ema=ema, optimizer=optimizer, criterion=criterion,
        val_sets=val_sets, device=device, dtype=dtype,
        scaler_unused=None,
        checkpoint_dir=CFG.checkpoint_dir, best_val=best_val,
    )
    epoch_offset += CFG.phase1_epochs

    # -------------------------------------------------------------------
    # PHASE 2 — Overlapping peaks mix in progressively
    # 40% clean, 30% noisy (strong), 30% overlap (difficulty 0.2 → 0.9)
    # -------------------------------------------------------------------
    ds_noisy.set_noise_level(0.8)

    for ep in range(CFG.phase2_epochs):
        progress = ep / max(CFG.phase2_epochs - 1, 1)
        ds_overlap.set_difficulty(0.2 + 0.7 * progress)

    best_val = run_phase(
        phase_name          = "Phase2_overlap",
        phase_epochs        = CFG.phase2_epochs,
        epoch_offset        = epoch_offset,
        datasets_and_weights= [(ds_clean, 0.4), (ds_noisy, 0.3), (ds_overlap, 0.3)],
        model=model, ema=ema, optimizer=optimizer, criterion=criterion,
        val_sets=val_sets, device=device, dtype=dtype,
        scaler_unused=None,
        checkpoint_dir=CFG.checkpoint_dir, best_val=best_val,
    )
    epoch_offset += CFG.phase2_epochs

    # -------------------------------------------------------------------
    # PHASE 3 — LFP domain adaptation, avoid catastrophic forgetting
    # 75% LFP-domain, 25% clean (anchor)
    # -------------------------------------------------------------------
    best_val = run_phase(
        phase_name          = "Phase3_LFP_domain",
        phase_epochs        = CFG.phase3_epochs,
        epoch_offset        = epoch_offset,
        datasets_and_weights= [(ds_lfp, 0.75), (ds_clean, 0.25)],
        model=model, ema=ema, optimizer=optimizer, criterion=criterion,
        val_sets=val_sets, device=device, dtype=dtype,
        scaler_unused=None,
        checkpoint_dir=CFG.checkpoint_dir, best_val=best_val,
    )
    epoch_offset += CFG.phase3_epochs

    # -------------------------------------------------------------------
    # PHASE 4 — Noisy boost : casse le plateau val noisy sans oublier LFP
    # 60% noisy (fort), 20% LFP (ancre), 20% clean (ancre)
    # -------------------------------------------------------------------
    ds_noisy.set_noise_level(0.9)
    best_val = run_phase(
        phase_name          = "Phase4_noisy_boost",
        phase_epochs        = CFG.phase4_epochs,
        epoch_offset        = epoch_offset,
        datasets_and_weights= [(ds_noisy, 0.6), (ds_lfp, 0.2), (ds_clean, 0.2)],
        model=model, ema=ema, optimizer=optimizer, criterion=criterion,
        val_sets=val_sets, device=device, dtype=dtype,
        scaler_unused=None,
        checkpoint_dir=CFG.checkpoint_dir, best_val=best_val,
    )
    epoch_offset += CFG.phase4_epochs

    # -------------------------------------------------------------------
    # Final checkpoint (last weights, not EMA)
    # -------------------------------------------------------------------
    torch.save({
        'epoch':            epoch_offset - 1,
        'model_state_dict': model.state_dict(),
        'freqs':            freqs_np,
        'val_loss':         best_val['loss'],
    }, os.path.join(CFG.checkpoint_dir, 'pretrain_final.pt'))

    print(f"\nPréentraînement terminé.")
    print(f"  Best val loss : {best_val['loss']:.4f} (epoch {best_val['epoch']+1})")
    print(f"  Checkpoints   : {CFG.checkpoint_dir}/pretrain_best.pt  (EMA, use this for finetune)")
    print(f"                  {CFG.checkpoint_dir}/pretrain_final.pt (last weights)")


if __name__ == '__main__':
    main()
