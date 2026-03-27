# NeuroSpec - Architecture Complete

## Overview

This document provides the complete NeuroSpec architecture, from raw LFP preprocessing to biophysical inference.
The model is organized around four axes:

- Axis 1: Bayesian posterior estimation with Normalizing Flows
- Axis 2: Temporal sequence modeling
- Axis 3: Spatial decomposition across channels
- Axis 4: Biophysical parameter decoding

The full stack remains compact (about 1.2M trainable parameters) to avoid severe overfitting on 92 Neuropixels sessions while keeping uncertainty-aware and biologically grounded outputs.

---

## Stage 0 - Input Format and Preprocessing

### Raw input

- Input tensor: `(B, C, T_raw)`
- `B`: batch size
- `C`: number of channels (up to 384 for Neuropixels)
- `T_raw`: raw time samples

### STFT to log-PSD

- Hann window length: `0.5s`
- Overlap: `75%`
- Frequency band: `1-45 Hz`
- Frequency resolution: `2 Hz` (adjustable)
- Number of bins: `F = 23`

Output tensor after STFT power spectrum:

- `(B, C, T_win, F)`

Implementation sketch:

```python
spec = torch.stft(x, n_fft=..., hop_length=..., window=torch.hann_window(...), return_complex=True)
psd = torch.pow(spec.abs(), 2)
```

### Normalization

- `log10` compression
- Per-channel z-score (computed on training data statistics)

This step is critical: without normalization, inter-channel impedance variability can dominate biological signal structure.

---

## Axis 3 - Spatial Decomposition

### Why spatial modeling is necessary

Neuropixels channels are not independent. A single oscillatory process (for example, hippocampal theta) can appear across dozens of channels simultaneously. Treating channels independently wastes statistical efficiency and forces redundant learning.

### Cross-channel self-attention

At each time window `t`, input slice is `(B, C, F)`. Multi-head self-attention is applied over the channel dimension:

```python
Q, K, V = Linear(F, d_enc)(x), Linear(F, d_enc)(x), Linear(F, d_enc)(x)
x_attn = Attention(Q, K, V)  # attention over channels C
```

Temporal stacking yields:

- `(B, C, T_win, d_enc)`

### Learned global/local decomposition

```python
x_global = x_attn.mean(dim=1, keepdim=True).expand_as(x_attn)
x_local  = x_attn - x_global
x_enc    = torch.cat([x_global, x_local], dim=-1)  # (B, C, T, 2*d_enc)
```

- `x_global`: shared spatial mode
- `x_local`: channel-specific residual

### Optional geometry-aware extension

If probe geometry is available (Neuropixels fixed geometry), replace vanilla channel attention with graph attention where edge weights reflect inter-channel distance:

- Candidate module: `torch_geometric.nn.GATConv`

This encourages biologically plausible locality (extracellular coupling decays with distance).

### Recommended hyperparameters

- Attention heads: `4`
- `d_enc`: `64`
- Dropout: `0.1`

---

## Axis 2 - Temporal Transformer

### Why temporal modeling is necessary

Aperiodic exponent and oscillatory components evolve smoothly over time. Independent frame-wise modeling discards temporal continuity and weakens identifiability under noise.

### Per-channel sequence encoder

For each channel `c`, model sequence `(B, T_win, d_model)` with shared weights across channels.

```python
pos_enc = SinusoidalPositionalEncoding(d_model=128, max_len=T_win)

layer = nn.TransformerEncoderLayer(
    d_model=128, nhead=4, dim_feedforward=256, dropout=0.1
)
transformer = nn.TransformerEncoder(layer, num_layers=3)
```

Output:

- `(B, C, T_win, 128)`

### Parameter sharing

The same temporal encoder is reused for all channels. This drastically reduces parameter count (especially for `C=384`) and enforces a universal representation of neural spectral dynamics.

### Long-sequence alternative

For very long recordings (`T_win` large), replace the transformer by a linear-complexity state-space alternative (for example Mamba) as an ablation.

---

## Physically Constrained Head

A lightweight MLP maps embeddings to raw parameters, then activations impose physical constraints.

### Head definition

```python
head = nn.Sequential(
    nn.Linear(128, 128),
    nn.GELU(),
    nn.Linear(128, P_raw),
)
```

### FOOOF parameterization and constraints

For `N_g = 6` peaks, total parameters per `(channel, time)`:

- `P = 2 + 3 * N_g = 20`

Constraint mapping:

- Offset: unconstrained (`identity`)
- Exponent `beta`: `beta > 0` (`softplus`)
- Peak center `f_k`: `1 < f_k < 45` (`sigmoid * 44 + 1`)
- Peak amplitude `a_k`: `a_k >= 0` (`softplus`)
- Peak width `sigma_k`: `sigma_k > 0` (`softplus`)

Peak ordering can be stabilized by sorted center frequencies (or a separation penalty when preferred).

---

## Axis 1 - Posterior with Normalizing Flow

### Why posterior modeling matters

Noisy or overlapping peaks can yield multiple plausible explanations. A point estimate cannot represent this ambiguity; a posterior can.

### Conditional Neural Spline Flow (NSF)

Use Rational-Quadratic spline couplings conditioned on temporal embeddings.

```python
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, RQSplineCouplingTransform

base_dist = StandardNormal(shape=[P])

transforms = CompositeTransform([
    RQSplineCouplingTransform(
        mask=alternating_mask(P),
        transform_net_create_fn=lambda in_f, out_f: ResidualNet(
            in_features=in_f + d_model,
            out_features=out_f,
            hidden_features=128,
        ),
        num_bins=8,
        tails="linear",
        tail_bound=3.0,
    )
    for _ in range(4)
])

flow = Flow(transform=transforms, distribution=base_dist)
```

### Inference protocol

- Draw `K=50` samples from `q(theta | x, t)`
- Decode each sample through analytical FOOOF decoder
- Return median estimate + 90% credible intervals

This provides calibrated uncertainty, especially useful for ambiguous spectral peaks.

---

## Analytical FOOOF Decoder (Fixed, Differentiable)

This decoder is non-learnable and enforces a structured, interpretable spectral reconstruction.

```python
def fooof_decode(theta: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    offset    = theta[..., 0:1]
    exponent  = theta[..., 1:2]
    peaks_f   = theta[..., 2::3]
    peaks_a   = theta[..., 3::3]
    peaks_bw  = theta[..., 4::3]

    log_f = torch.log10(freqs.view(1, 1, 1, -1) + 1e-6)
    aperiodic = offset - exponent * log_f

    f_exp   = freqs.view(1, 1, 1, 1, -1)
    cf_exp  = peaks_f.unsqueeze(-1)
    amp_exp = peaks_a.unsqueeze(-1)
    bw_exp  = peaks_bw.unsqueeze(-1)

    gaussians = amp_exp * torch.exp(-0.5 * ((f_exp - cf_exp) / bw_exp) ** 2)
    periodic = gaussians.sum(dim=-2)

    return aperiodic + periodic
```

Key properties:

- Zero trainable parameters
- Exact analytical gradients
- Strong interpretability
- Reduced overfitting risk on spectral shapes

---

## Axis 4 - Biophysical Chain

Maps spectral parameters to biologically meaningful latent quantities.

### Relation 1: Aperiodic exponent to E/I ratio

Analytical relation:

`beta ~= 2 - log2(r_EI)`

Differentiable mapping:

```python
r_EI = 2 ** (2 - beta)
```

### Relation 2: Offset to global activity proxy

```python
activity = 10 ** offset
```

### Relation 3: Time constants via knee-frequency proxy

Using Lorentzian structure:

`S(f) = A / (1 + (f / f_knee)^2)`, and `tau = 1 / (2*pi*f_knee)`

To separate excitation/inhibition timescales, use constrained learning from spectral context with biological priors.

### Decoder sketch

```python
class BiophysicalDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tau_head = nn.Sequential(
            nn.Linear(3, 32),
            nn.Softplus(),
            nn.Linear(32, 2),
            nn.Softplus(),
        )

    def forward(self, theta):
        exponent = theta[..., 1]
        offset   = theta[..., 0]
        peaks_f  = theta[..., 2::3]

        r_EI = 2 ** (2 - exponent)
        activity = 10 ** offset

        tau_input = torch.stack([exponent, offset, peaks_f.mean(-1)], dim=-1)
        taus = self.tau_head(tau_input) * 1e-3

        return torch.stack([r_EI, activity, taus[..., 0], taus[..., 1]], dim=-1)
```

Expected biological trend (example hypothesis):

- In older MitoPark mice: steeper aperiodic slopes (`beta` up) and lower inferred `r_EI`, consistent with inhibition/excitation imbalance in neurodegeneration contexts.

---

## Composite Loss Function

Total loss:

`L = L_recon + L_sparse + L_smooth + L_spatial + beta_ELBO * L_ELBO + L_bio`

### Terms and rationale

- `L_recon`: Huber reconstruction (`delta=0.5`) for robustness to movement artifacts
- `L_sparse`: L1 on peak amplitudes to discourage unnecessary peaks
- `L_smooth`: temporal smoothness on consecutive parameter states
- `L_spatial`: covariance consistency with pretrained spatial prior
- `L_ELBO`: variational objective for the flow posterior
- `L_bio`: consistency between predicted biophysics and analytical constraints

KL annealing:

- `beta_ELBO` increases from `0` to `1` over the first `100` epochs to reduce posterior collapse risk.

---

## Three-Phase Training Curriculum

### Phase 1 - Synthetic pretraining

- Large simulated corpus (`~5M` spectra)
- Exact parameter supervision
- Objective emphasis: `L_ELBO + L_recon`

### Phase 2 - Unsupervised real-data fine-tuning

- 92 Neuropixels sessions
- Objective: `L_recon + L_sparse + L_smooth`
- Reduced learning rate (`x0.1` versus pretraining)

### Phase 3 - Biophysical constrained fine-tuning

- Add `L_bio + L_spatial`
- Freeze temporal and spatial encoders
- Update constrained head + biophysical chain

This preserves sim-to-real representations while tuning biological consistency.

---

## Recommended Hyperparameters

| Component | Parameter | Recommended value |
|---|---|---|
| STFT | Window length | 0.5s |
| STFT | Overlap | 75% |
| Spatial encoder | Attention heads | 4 |
| Spatial encoder | `d_enc` | 64 |
| Temporal encoder | Layers | 3 |
| Temporal encoder | `d_model` | 128 |
| Physics head | Max peaks (`N_peaks`) | 6 |
| NSF | Coupling layers | 4 |
| NSF | Spline bins | 8 |
| Biophys head | MLP layers | 2 |
| Loss | `lambda_1` (sparsity) | 0.05 |
| Loss | `lambda_2` (smoothness) | 0.01 |
| Loss | `lambda_3` (spatial) | 0.001 |
| Loss | `lambda_4` (biophysical) | 0.1 |
| ELBO | final `beta_ELBO` | 1.0 |
| Optimization | Optimizer | AdamW (`lr=3e-4`) |
| Optimization | Batch size | 32 sessions |

---

## Practical Notes After Pretraining

If pretraining is complete, the next operational stage is:

1. Real-data fine-tuning (Phase 2)
2. Biophysical constrained fine-tuning (Phase 3)
3. Evaluation and statistical validation on MitoPark vs controls

The architecture above is designed to keep interpretability, uncertainty quantification, and biological plausibility jointly optimized.
