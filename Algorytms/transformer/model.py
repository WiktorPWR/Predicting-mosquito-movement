"""Small transformer encoders predicting the next 10 displacements at once (PyTorch, CPU).

Literature basis:
  - Giuliari et al., "Transformer Networks for Trajectory Forecasting", ICPR 2020 (arXiv:2003.08111): a plain
    transformer without interaction modelling was best on TrajNet, so a plain encoder is enough to start with.
  - AeroCast, Transformer-MDN for 3D aerial trajectories (arXiv:2606.25122): 20 steps of 3D displacements as
    input, Pre-LN encoder, the last frame's token goes to the head.
  - Nie et al., PatchTST, ICLR 2023 (arXiv:2211.14730): patching (several consecutive steps = one token) improves
    transformers and allows longer windows (variant T3).

Input, output and rollout are those of the selected GRU / LSTM (Algorytms/lstm/model.py):
  - input: the last `window` frame-to-frame displacements / training std (x 0.349, y 0.250, z 0.434 cm),
    shape (batch, window, 3);
  - output: the next 10 displacements (0.1 s) at once, shape (batch, 10, 3); positions = last position +
    cumulative sum;
  - forecasts beyond 0.1 s: the 10 predicted positions are appended to the window, the 10 oldest dropped, and the
    network is called again.

Network:
  - patching: (batch, window, 3) -> (batch, window / patch, 3 * patch), one token = `patch` consecutive frames;
  - Linear embedding + learned positional encoding;
  - Pre-LN encoder (nn.TransformerEncoderLayer, norm_first=True, dropout 0.1), then a final LayerNorm;
  - Linear(d -> 30) on the last token -> (batch, 10, 3).
"""
import torch
from torch import nn

from Algorytms.lstm.model import STEPS, Normaliser, Predictor

VARIANTS = {
    "tf_t1": dict(d=20, ff=40, heads=2, layers=1, window=20, patch=1, label="Transformer T1 (d 20)"),
    "tf_t2": dict(d=32, ff=64, heads=2, layers=1, window=20, patch=1, label="Transformer T2 (d 32)"),
    "tf_t3": dict(d=20, ff=40, heads=2, layers=1, window=100, patch=10, label="Transformer T3 (window 100, patch 10)"),
}
SEEDS = range(5)
ARCH_KEYS = ("d", "ff", "heads", "layers", "window", "patch")


def arch(variant):
    return {k: VARIANTS[variant][k] for k in ARCH_KEYS}


class KomarTF(nn.Module):
    def __init__(self, d=20, ff=40, heads=2, layers=1, window=20, patch=1, steps=STEPS):
        super().__init__()
        assert d % heads == 0, "the number of heads must divide d"
        assert window % patch == 0, "the window must be divisible by the patch length"
        self.arch = dict(d=d, ff=ff, heads=heads, layers=layers, window=window, patch=patch)
        self.patch, self.steps = patch, steps
        self.embed = nn.Linear(3 * patch, d)
        self.pos = nn.Parameter(torch.randn(1, window // patch, d) * 0.02)     # learned positional encoding
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=ff, dropout=0.1,
                                           batch_first=True, norm_first=True)  # Pre-LN
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, steps * 3)

    def forward(self, x):
        """(batch, window, 3) -> (batch, steps, 3)."""
        B = x.shape[0]
        x = x.reshape(B, -1, 3 * self.patch)            # patching: consecutive frames glued into one token
        z = self.embed(x) + self.pos
        z = self.norm(self.encoder(z))
        return self.head(z[:, -1]).view(B, -1, 3)       # last token -> the next `steps` displacements


class TFPredictor(Predictor):
    """Benchmark interface; the rollout (forecast) is the one of the GRU / LSTM study."""

    def __init__(self, net, norm, name, label):
        self.net, self.norm, self.inp, self.window = net.eval(), norm, "disp", net.arch["window"]
        self.name, self.label = name, label
        self.params = dict(model="transformer", **net.arch, steps=net.steps, **norm.to_dict())


def save(path, net, norm, history):
    torch.save({"state": net.state_dict(), "arch": net.arch, "steps": net.steps, "norm": norm.to_dict(),
                "history": history}, path)


def load(path, name, label):
    ck = torch.load(path, weights_only=False)
    net = KomarTF(**ck["arch"], steps=ck["steps"])
    net.load_state_dict(ck["state"])
    return TFPredictor(net, Normaliser(**ck["norm"]), name, label)
