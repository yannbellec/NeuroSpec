"""
DiffFOOOF — Differentiable FOOOF
Neural surrogate for spectral parameterization.

Golden version (v3):
  - softmax gaps for CFs (ordered, bounded, smooth gradients)
  - no hard clamps, justified differentiable constraints
  - DiffFOOOFLoss: FULL parameter supervision with SOFT MATCHING
      Fixes vs v2:
      [1] Positional slot assignment replaced by soft distance-based matching.
          Model peaks and GT peaks are matched by minimum CF distance,
          not by slot index — correct gradients even when peaks reorder.
      [2] BCE on softplus amplitudes removed (mathematically incoherent:
          softplus ∈ ℝ⁺, BCE expects logits ∈ ℝ).
          Replaced by masked L1 on unmatched model slots → pushes unused
          peaks to zero without gradient inconsistency.
      [3] bw penalty is now one-sided: only penalises bw > bw_soft_max
          (default 4 Hz). Narrow peaks are not penalised, eliminating
          the systematic bias toward artificially thin peaks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Soft peak matching
# ---------------------------------------------------------------------------

def soft_match_peaks(pred_cfs, gt_cfs, gt_mask):
    """
    Match predicted peaks to ground-truth peaks by minimum CF distance.
    Fully differentiable. No scipy. O(B * K²).

    Args:
        pred_cfs : [B, K]   model centre frequencies (ordered, all slots active)
        gt_cfs   : [B, K]   ground-truth CFs (sorted ascending, padded with 0)
        gt_mask  : [B, K]   float — 1.0 for real GT peaks, 0.0 for padding

    Returns:
        match_idx     : [B, K_gt]  long  — for each GT peak, which pred slot it matched
        matched_mask  : [B, K]    float — 1.0 for pred slots that were matched to a GT peak
        unmatched_mask: [B, K]    float — 1.0 for pred slots with no GT peak assigned

    Strategy (greedy nearest-neighbour, ascending CF order):
        For each GT peak (in CF order), assign it to the closest unoccupied pred slot.
        GT padding slots (mask=0) are skipped — they consume no pred slot.
        Greedy works well here because K is small (6) and CFs are ordered on both sides.
    """
    B, K = pred_cfs.shape
    device = pred_cfs.device

    # Distance matrix [B, K_gt, K_pred] — |gt_cf - pred_cf|
    dist = (gt_cfs.unsqueeze(2) - pred_cfs.unsqueeze(1)).abs()   # [B, K, K]

    match_idx      = torch.full((B, K), -1, dtype=torch.long, device=device)
    slot_used      = torch.zeros(B, K, dtype=torch.bool, device=device)

    # Greedy assignment — iterate over GT slots (small K, no perf issue)
    n_gt = K
    for j in range(n_gt):
        active_gt = gt_mask[:, j] > 0.5                          # [B] bool
        if not active_gt.any():
            continue
        d_j = dist[:, j, :].clone()                              # [B, K]
        # Mask already-used pred slots with large distance
        d_j[slot_used] = 1e9
        best_slot = d_j.argmin(dim=1)                            # [B]
        # Only assign for samples where this GT peak exists
        for b in range(B):
            if active_gt[b]:
                match_idx[b, j] = best_slot[b]
                slot_used[b, best_slot[b]] = True

    # Build matched_mask: which pred slots received a GT peak assignment
    matched_mask   = slot_used.float()                            # [B, K]
    unmatched_mask = 1.0 - matched_mask                          # [B, K]

    return match_idx, matched_mask, unmatched_mask


class DiffFOOOF(nn.Module):
    """
    Differentiable spectral parameterization model.

    Learns to decompose a PSD into:
        - Aperiodic component : offset - exponent * log10(f)
        - Periodic component  : sum of K Gaussians (cf, amp, bw)

    The analytical decoder is fixed — no learned parameters in decoding.
    All physical constraints are enforced via smooth differentiable functions.

    Architecture unchanged from golden version. n_peaks fixed at 6.
    """

    def __init__(self, freqs, n_peaks=6, hidden_dim=256):
        super().__init__()
        self.register_buffer('freqs', torch.tensor(freqs, dtype=torch.float32))
        self.n_peaks   = n_peaks
        self.f_min     = float(freqs[0])
        self.f_max     = float(freqs[-1])
        self.bandwidth = self.f_max - self.f_min

        # Physical bounds for bandwidth
        self.bw_min = 0.5
        self.bw_max = 8.0

        # Output layout: 2 (aperiodic) + K (amps) + K (bws) + (K+1) (gaps for CFs)
        self.out_dim = 2 + n_peaks + n_peaks + (n_peaks + 1)

        self.encoder = nn.Sequential(
            nn.Linear(len(freqs), hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.out_dim),
        )

    def forward(self, psd_input):
        """
        Args:
            psd_input : [B, F]  log10-transformed PSD

        Returns:
            pred_psd    : [B, F]  reconstructed PSD
            params_dict : dict of constrained parameters
                'offset'   [B, 1]
                'exponent' [B, 1]
                'amps'     [B, K]
                'bws'      [B, K]
                'cfs'      [B, K]
        """
        raw = self.encoder(psd_input)

        raw_offset = raw[:, 0:1]
        raw_exp    = raw[:, 1:2]
        raw_amps   = raw[:, 2 : 2 + self.n_peaks]
        raw_bws    = raw[:, 2 + self.n_peaks : 2 + 2 * self.n_peaks]
        raw_gaps   = raw[:, 2 + 2 * self.n_peaks :]           # [B, K+1]

        # --- Soft physical constraints ---

        # offset : free — network learns its own scale
        offset = raw_offset

        # exponent : softplus ensures > 0  (physiological 1/f slope)
        exponent = F.softplus(raw_exp)

        # amps : softplus ensures >= 0 ; L1/sparsity loss handles zeros
        amps = F.softplus(raw_amps)

        # bws : sigmoid maps to [bw_min, bw_max]
        bws = self.bw_min + (self.bw_max - self.bw_min) * torch.sigmoid(raw_bws)

        # CFs : softmax gaps — ordered, bounded, smooth gradients everywhere
        # K+1 raw values → K+1 fractions summing to 1 → K ordered CFs in [f_min, f_max]
        gaps_pct = F.softmax(raw_gaps, dim=-1)                # [B, K+1]
        gaps_hz  = gaps_pct * self.bandwidth                  # [B, K+1]
        cfs      = self.f_min + torch.cumsum(gaps_hz[:, :-1], dim=-1)  # [B, K]

        # --- Fixed analytical decoder (no learned params) ---
        log_f = torch.log10(self.freqs + 1e-6).unsqueeze(0)  # [1, F]

        # Aperiodic : power-law in log-freq space
        aperiodic = offset - exponent * log_f                 # [B, F]

        # Periodic : sum of K Gaussians
        freqs_exp = self.freqs.view(1, 1, -1)                 # [1, 1, F]
        cfs_exp   = cfs.unsqueeze(2)                          # [B, K, 1]
        bws_exp   = bws.unsqueeze(2)                          # [B, K, 1]
        amps_exp  = amps.unsqueeze(2)                         # [B, K, 1]

        gaussians = amps_exp * torch.exp(
            -0.5 * ((freqs_exp - cfs_exp) / bws_exp) ** 2
        )                                                     # [B, K, F]
        periodic  = gaussians.sum(dim=1)                      # [B, F]

        pred_psd = aperiodic + periodic

        params_dict = {
            'offset':   offset,
            'exponent': exponent,
            'cfs':      cfs,
            'amps':     amps,
            'bws':      bws,
        }
        return pred_psd, params_dict


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class DiffFOOOFLoss(nn.Module):
    """
    Composite loss for DiffFOOOF.

    Reconstruction terms (always active):
        L_recon      : Huber(PSD_pred, PSD_true)    — robust to EEG artifacts
        L_sparse     : L1 on all amps               — global sparsity prior
        L_bw_excess  : one-sided penalty on bw > bw_soft_max
                       — prevents degenerate wide Gaussians WITHOUT biasing
                         toward artificially narrow peaks (v2 bug fixed)

    Supervised parameter terms (active when gt_params provided, i.e. pretraining):
        L_ap          : MSE on offset + exponent          — aperiodic supervision
        L_peaks       : matched MSE on (cfs, amps, bws)   — soft distance-based matching
        L_unmatched   : L1 on amps of unmatched pred slots — replaces BCE (v2 bug fixed)

    Peak matching (fix for v2 positional assignment bug):
        Uses soft_match_peaks() — greedy nearest-neighbour on CF distance.
        Model slot i is paired with the closest GT peak, not the i-th GT peak.
        This gives correct gradients even when peaks reorder or partially overlap.
        Unmatched model slots (no GT peak assigned) are pushed toward amp=0 via L1.

    gt_params expected keys:
        'offset'     [B]     aperiodic offset
        'exponent'   [B]     aperiodic exponent
        'cfs'        [B, K]  peak centre frequencies (sorted, padded with 0)
        'amps'       [B, K]  peak amplitudes         (padded with 0)
        'bws'        [B, K]  peak bandwidths         (padded with 0)
        'peak_mask'  [B, K]  1.0 for real peaks, 0.0 for padding
    """

    def __init__(
        self,
        lambda_sparse    : float = 0.10,
        lambda_bw_excess : float = 0.05,   # one-sided bw penalty (replaces L2 bw)
        bw_soft_max      : float = 4.0,    # Hz — only penalise bw above this
        lambda_ap        : float = 0.50,
        lambda_peaks     : float = 0.30,
        lambda_unmatched : float = 0.10,   # L1 on unmatched pred slots (replaces BCE)
    ):
        super().__init__()
        self.lambda_sparse    = lambda_sparse
        self.lambda_bw_excess = lambda_bw_excess
        self.bw_soft_max      = bw_soft_max
        self.lambda_ap        = lambda_ap
        self.lambda_peaks     = lambda_peaks
        self.lambda_unmatched = lambda_unmatched

    def forward(self, pred_psd, true_psd, params_dict, gt_params=None):
        """
        Args:
            pred_psd    : [B, F]
            true_psd    : [B, F]
            params_dict : dict from DiffFOOOF.forward()
            gt_params   : optional dict — see class docstring

        Returns:
            total_loss : scalar
            loss_dict  : dict of float scalars for logging
        """
        amps = params_dict['amps']   # [B, K]
        bws  = params_dict['bws']    # [B, K]

        # --- Reconstruction ---
        l_recon  = F.huber_loss(pred_psd, true_psd, delta=1.0)

        # Global sparsity — L1 on all amplitudes, always active
        l_sparse = amps.mean()

        # One-sided bw excess penalty: relu(bw - bw_soft_max)²
        # Zero for bw ≤ bw_soft_max, quadratic above — no bias toward narrow peaks
        bw_excess   = F.relu(bws - self.bw_soft_max)
        l_bw_excess = (bw_excess ** 2).mean()

        total = (
            l_recon
            + self.lambda_sparse    * l_sparse
            + self.lambda_bw_excess * l_bw_excess
        )

        loss_dict = {
            'recon'      : l_recon.item(),
            'sparse'     : l_sparse.item(),
            'bw_excess'  : l_bw_excess.item(),
            'ap'         : 0.0,
            'peaks'      : 0.0,
            'unmatched'  : 0.0,
        }

        if gt_params is None:
            return total, loss_dict

        # --- Aperiodic supervision ---
        l_ap = (
            F.mse_loss(params_dict['exponent'].squeeze(1), gt_params['exponent'])
            + F.mse_loss(params_dict['offset'].squeeze(1),   gt_params['offset'])
        )

        # --- Soft peak matching ---
        # match_idx      [B, K]: for each GT slot j, which pred slot was assigned (-1 if GT absent)
        # matched_mask   [B, K]: 1.0 for pred slots that received a GT assignment
        # unmatched_mask [B, K]: 1.0 for pred slots with no GT peak (should → amp~0)
        gt_mask = gt_params['peak_mask']                         # [B, K]
        match_idx, matched_mask, unmatched_mask = soft_match_peaks(
            params_dict['cfs'], gt_params['cfs'], gt_mask
        )

        # Gather matched pred params using match_idx
        # For each GT peak j, retrieve the pred slot it was matched to
        B, K = amps.shape
        device = amps.device
        n_real = gt_mask.sum().clamp(min=1.0)

        # Build matched pred tensors [B, K] — only GT-active slots contribute
        pred_cfs_matched  = torch.zeros(B, K, device=device)
        pred_amps_matched = torch.zeros(B, K, device=device)
        pred_bws_matched  = torch.zeros(B, K, device=device)

        for j in range(K):
            active = gt_mask[:, j] > 0.5                        # [B] bool
            if not active.any():
                continue
            idx_j = match_idx[:, j].clamp(min=0)                # [B], safe gather
            # Gather from pred using assigned slot index
            pred_cfs_matched[:, j]  = params_dict['cfs'].gather(1, idx_j.unsqueeze(1)).squeeze(1)
            pred_amps_matched[:, j] = amps.gather(1, idx_j.unsqueeze(1)).squeeze(1)
            pred_bws_matched[:, j]  = bws.gather(1, idx_j.unsqueeze(1)).squeeze(1)

        # Masked MSE: only over GT-active slots
        def masked_mse(pred, target, mask, n):
            return (F.mse_loss(pred, target, reduction='none') * mask).sum() / n

        l_cfs  = masked_mse(pred_cfs_matched,  gt_params['cfs'],  gt_mask, n_real)
        l_amps = masked_mse(pred_amps_matched, gt_params['amps'], gt_mask, n_real)
        l_bws_ = masked_mse(pred_bws_matched,  gt_params['bws'],  gt_mask, n_real)
        l_peaks = l_cfs + l_amps + l_bws_

        # --- Unmatched pred slots: push amp toward 0 via L1 (not BCE) ---
        # amps ∈ ℝ⁺ (softplus) — L1 is coherent, BCE would not be
        l_unmatched = (amps * unmatched_mask).sum() / unmatched_mask.sum().clamp(min=1.0)

        total = (
            total
            + self.lambda_ap        * l_ap
            + self.lambda_peaks     * l_peaks
            + self.lambda_unmatched * l_unmatched
        )

        loss_dict['ap']        = l_ap.item()
        loss_dict['peaks']     = l_peaks.item()
        loss_dict['unmatched'] = l_unmatched.item()

        return total, loss_dict
