"""PCR + complex screening — Path-X model (verbatim port).

Model code ported as-is from the PCT bench trainer
(phase8_experiment/src/train_with_checkpoint.py, cell="pcr").
Self-contained: torch is the only dependency.

Reference result (Long Range Arena Path-X, rule-compliant, test n=20,000):
    92.71 +/- 0.89 over 3 seeds; best seed 93.50 (weights in ../model/).
See ../README.md for the mathematical documentation.
"""

import torch

# ── R-EX3: PCR (phase-coherent recurrence) ────────────────────────────────
# LRU-style complex diagonal LTI recurrence (Orvieto et al. 2023) applied via
# FFT convolution; bidirectional; ring init of |lambda| near 1 with RESTRICTED
# PHASE init and gamma normalization (both named crucial for Path-X in the LRU
# paper). Complex eigenvalues = input-independent phase rotation per step =
# coherent transport (continuous RoPE); input-dependent gates stay real.
# Self-contained in this file so the DOK chain's scp delivers the model.


class PCRLayer(torch.nn.Module):
    def __init__(self, d_model, d_state=None, r_min=0.999, r_max=0.9999,
                 max_phase=0.06283185307, real_eigs=False, bidirectional=True):
        super().__init__()
        import math as _m
        N = d_state or d_model
        self.N = N
        self.bidirectional = bidirectional
        self.real_eigs = real_eigs
        n_dir = 2 if bidirectional else 1
        u = torch.rand(n_dir, N)
        nu = torch.log(-0.5 * torch.log(u * (r_max**2 - r_min**2) + r_min**2))
        self.nu_log = torch.nn.Parameter(nu)
        theta = torch.rand(n_dir, N) * max_phase
        self.theta_log = torch.nn.Parameter(torch.log(theta.clamp_min(1e-4)))
        self.B_re = torch.nn.Parameter(torch.randn(n_dir, d_model, N) / _m.sqrt(2 * d_model))
        self.B_im = torch.nn.Parameter(torch.randn(n_dir, d_model, N) / _m.sqrt(2 * d_model))
        self.C_re = torch.nn.Parameter(torch.randn(n_dir, N, d_model) / _m.sqrt(N))
        self.C_im = torch.nn.Parameter(torch.randn(n_dir, N, d_model) / _m.sqrt(N))
        self.skip_D = torch.nn.Parameter(torch.randn(d_model) * 0.01)

    def _kernel(self, L, direction):
        log_mag = -self.nu_log[direction].exp()
        if self.real_eigs:
            phase = torch.zeros_like(log_mag)
        else:
            phase = self.theta_log[direction].exp()
        t = torch.arange(L, device=log_mag.device, dtype=torch.float32)
        mag = (t[None, :] * log_mag[:, None]).exp()
        ang = t[None, :] * phase[:, None]
        gamma = (1 - (2 * log_mag).exp()).clamp_min(1e-6).sqrt()
        k = torch.complex(mag * torch.cos(ang), mag * torch.sin(ang))
        return k * gamma[:, None].to(k.dtype)

    def _run_dir(self, x, direction):
        B_, L, _ = x.shape
        Bc = torch.complex(self.B_re[direction], self.B_im[direction])
        u = torch.einsum("bld,dn->bln", x.to(torch.complex64), Bc)
        k = self._kernel(L, direction)
        n_fft = 2 * L
        Uf = torch.fft.fft(u.transpose(1, 2), n=n_fft)
        Kf = torch.fft.fft(k, n=n_fft)
        y = torch.fft.ifft(Uf * Kf[None], n=n_fft)[..., :L]
        Cc = torch.complex(self.C_re[direction], self.C_im[direction])
        return torch.einsum("bnl,nd->bld", y, Cc).real

    def forward(self, x):
        y = self._run_dir(x, 0)
        if self.bidirectional:
            y = y + self._run_dir(x.flip(1), 1).flip(1)
        return y + self.skip_D[None, None, :] * x

class _SeqBatchNorm(torch.nn.Module):
    """BatchNorm1d over the channel dim of (B, L, D) sequences.

    LRU (Orvieto et al. 2023) App. D.1: "we always use batch normalization";
    S5's Path-X config likewise sets batchnorm=True. PyTorch default
    momentum=0.1 equals S5's bn_momentum=0.9 (JAX convention is inverted).
    """

    def __init__(self, d_model):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(d_model)

    def forward(self, x):
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


def _make_norm(norm, d_model):
    if norm == "batch":
        return _SeqBatchNorm(d_model)
    if norm == "layer":
        return torch.nn.LayerNorm(d_model)
    raise ValueError(f"unknown pcr norm: {norm!r}")


class PCRBlock(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, norm="layer", block="glu_ffn", **kw):
        super().__init__()
        nn = torch.nn
        self.block_kind = block
        self.ln1 = _make_norm(norm, d_model)
        self.pcr = PCRLayer(d_model, **kw)
        self.drop = nn.Dropout(dropout)
        if block == "s5":
            # LRU App. D.1 Path-X mixing = S5's half_glu2: GELU on the SSM
            # output, sigmoid gate from one linear, NO extra linear on the
            # value path and NO separate FFN sublayer.
            self.gate = nn.Linear(d_model, d_model)
        elif block == "glu_ffn":  # v2 block (kept for ablation/compat)
            self.glu = nn.Linear(d_model, 2 * d_model)
            self.ln2 = _make_norm(norm, d_model)
            self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(4 * d_model, d_model))
        else:
            raise ValueError(f"unknown pcr block: {block!r}")

    def forward(self, x):
        h = self.pcr(self.ln1(x))
        if self.block_kind == "s5":
            h1 = self.drop(torch.nn.functional.gelu(h))
            return x + self.drop(h1 * torch.sigmoid(self.gate(h1)))
        a, b = self.glu(h).chunk(2, dim=-1)
        x = x + self.drop(a * torch.sigmoid(b))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class ComplexScreenBlock(torch.nn.Module):
    """Chunked complex screening attention block for the PCR hybrid (R-EX4).

    Paper-faithful screening gate ported to (re,im) pair tensors on a REAL
    residual stream: L2-normalized complex q,k -> s = Re<q̄,k̄> -> trim-and-
    square alpha = r^2 * relu(s - t)^2 (per-head learnable width, NO row
    normalization = non-competing), complex v aggregation, TanhNorm magnitude
    regulator, modReLU(g) Hadamard gate. No softmask (Phase 14 audit: defect).
    Attention is local within non-overlapping chunks; global transport is the
    PCR recurrence's job (phase-coherent counterpart of MEGA-chunk).
    """

    def __init__(self, d_model, heads=4, dim_head=32, chunk=1024,
                 dropout=0.0, norm="batch", s_r_init=-3.0, eps=1e-8):
        super().__init__()
        nn = torch.nn
        self.heads, self.dim_head, self.chunk, self.eps = heads, dim_head, chunk, eps
        inner = heads * dim_head
        self.norm = _make_norm(norm, d_model)
        # complex linear dim -> 4*inner as re/im pair mats
        self.qkvg_re = nn.Linear(d_model, 4 * inner)
        self.qkvg_im = nn.Linear(d_model, 4 * inner)
        self.s_r = nn.Parameter(torch.full((heads,), float(s_r_init)))
        self.modrelu_b = nn.Parameter(torch.zeros(1))
        self.out = nn.Linear(2 * inner, d_model)
        self.drop = nn.Dropout(dropout)

    def _split(self, t):
        B, L, _ = t.shape
        return t.view(B, L // self.chunk, self.chunk, self.heads, self.dim_head).permute(0, 1, 3, 2, 4)

    def forward(self, x):
        B, L, _ = x.shape
        h = self.norm(x)
        pr, pi = self.qkvg_re(h), self.qkvg_im(h)
        q_re, k_re, v_re, g_re = (self._split(t) for t in pr.chunk(4, dim=-1))
        q_im, k_im, v_im, g_im = (self._split(t) for t in pi.chunk(4, dim=-1))
        # complex L2 norm of q, k
        qn = (q_re.pow(2) + q_im.pow(2)).sum(-1, keepdim=True).clamp_min(self.eps).sqrt()
        kn = (k_re.pow(2) + k_im.pow(2)).sum(-1, keepdim=True).clamp_min(self.eps).sqrt()
        q_re, q_im, k_re, k_im = q_re / qn, q_im / qn, k_re / kn, k_im / kn
        # s = Re<q̄_i, k_j> = qre·kre + qim·kim  (within chunk)
        s = torch.einsum("bchid,bchjd->bchij", q_re, k_re) + \
            torch.einsum("bchid,bchjd->bchij", q_im, k_im)
        r = torch.exp(self.s_r) + 1.0
        t = (1.0 - 1.0 / r).view(1, 1, self.heads, 1, 1)
        alpha = (r * r).view(1, 1, self.heads, 1, 1) * torch.nn.functional.relu(s - t).pow(2)
        # aggregate complex v, NO row-norm
        u_re = torch.einsum("bchij,bchjd->bchid", alpha, v_re)
        u_im = torch.einsum("bchij,bchjd->bchid", alpha, v_im)
        # TanhNorm on complex magnitude
        mag = (u_re.pow(2) + u_im.pow(2)).sum(-1, keepdim=True).clamp_min(self.eps).sqrt()
        scale = torch.tanh(mag) / mag
        u_re, u_im = u_re * scale, u_im * scale
        # modReLU gate + complex Hadamard
        gmag = (g_re.pow(2) + g_im.pow(2)).clamp_min(self.eps).sqrt()
        gs = torch.nn.functional.relu(gmag + self.modrelu_b) / gmag
        g_re, g_im = g_re * gs, g_im * gs
        o_re = u_re * g_re - u_im * g_im
        o_im = u_re * g_im + u_im * g_re
        o = torch.cat([o_re, o_im], dim=-1).permute(0, 1, 3, 2, 4).reshape(B, L, -1)
        return x + self.drop(self.out(o))


class PCRClassifier(torch.nn.Module):
    def __init__(self, vocab=256, num_classes=2, d_model=128, n_layers=6,
                 d_state=None, norm="layer", block="glu_ffn", encoder="embed",
                 dropout=0.1, real_eigs=False, r_min=0.999, r_max=0.9999,
                 max_phase=0.06283185307,
                 screen_layers=(), screen_heads=4, screen_dim_head=32,
                 screen_chunk=1024):
        super().__init__()
        nn = torch.nn
        self.encoder_kind = encoder
        if encoder == "linear":
            # S4/S5/LRU LRA convention: scalar pixel -> dense encoder
            self.embed = nn.Linear(1, d_model)
        elif encoder == "embed":
            self.embed = nn.Embedding(vocab, d_model)
        else:
            raise ValueError(f"unknown pcr encoder: {encoder!r}")
        blocks = []
        for i in range(n_layers):
            blocks.append(
                PCRBlock(d_model, dropout=dropout, norm=norm, block=block,
                         d_state=d_state, real_eigs=real_eigs,
                         r_min=r_min, r_max=r_max, max_phase=max_phase))
            if i in set(screen_layers):
                # R-EX4 hybrid: complex screening attention (selection)
                # interleaved with PCR (transport)
                blocks.append(ComplexScreenBlock(
                    d_model, heads=screen_heads, dim_head=screen_dim_head,
                    chunk=screen_chunk, dropout=dropout, norm=norm))
        self.blocks = nn.ModuleList(blocks)
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, tokens):
        if self.encoder_kind == "linear":
            x = self.embed(tokens.float().unsqueeze(-1) / 255.0)
        else:
            x = self.embed(tokens)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_out(x).mean(dim=1))


def build_pcr_classifier(seq_len, vocab, lr_rec_factor=0.25, lr_bc_factor=None,
                         final_eval_sweep=None, eval_split=None, **cfg):
    # lr_* factors are consumed by the optimizer setup; final_eval_sweep /
    # eval_split by the final-sweep branch in main() — none reach the model.
    return PCRClassifier(vocab=vocab, **cfg)

def pcr_param_groups(model, lr, weight_decay, lr_rec_factor=0.25,
                     lr_bc_factor=None):
    """Optimizer groups for cell=pcr.

    lr_bc_factor=None (v2 / LRU App. D.2 reading): A(nu,theta), B, skip_D at
    lr*lr_rec_factor wd=0; C and the rest at base lr with wd.
    lr_bc_factor set (v3 / S5 opt_config=BandCdecay): A(nu,theta), skip_D at
    lr*lr_rec_factor wd=0; B and C at lr*lr_bc_factor WITH weight decay.
    """
    lam_keys = ("nu_log", "theta_log", "skip_D")
    b_keys = ("B_re", "B_im")
    c_keys = ("C_re", "C_im")
    lam, bc, other = [], [], []
    for n, p in model.named_parameters():
        if any(k in n for k in lam_keys):
            lam.append(p)
        elif any(k in n for k in b_keys):
            (bc if lr_bc_factor is not None else lam).append(p)
        elif any(k in n for k in c_keys):
            (bc if lr_bc_factor is not None else other).append(p)
        else:
            other.append(p)
    groups = [
        {"params": other, "lr": lr, "weight_decay": weight_decay},
        {"params": lam, "lr": lr * lr_rec_factor, "weight_decay": 0.0},
    ]
    if bc:
        groups.append({"params": bc, "lr": lr * lr_bc_factor,
                       "weight_decay": weight_decay})
    return groups
