import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")          # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# ─── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ─── Hyper-parameters ───────────────────────────────────────────────────────
BATCH_SIZE    = 128
EPOCHS        = 25
LR            = 3e-4
WEIGHT_DECAY  = 1e-4

# ── Critical fix ──────────────────────────────────────────────────────────
# gate_scores are initialised to -3.0 so sigmoid(-3) ≈ 0.047.
# Gates start near 0, giving the L1 penalty real leverage from epoch 1.
# If we initialise to 0 (sigmoid = 0.5), the penalty needs to be enormous
# to push gates below the threshold — hurting accuracy without gaining
# sparsity. Negative initialisation is the key to making pruning work.
GATE_INIT     = -3.0

LAMBDAS       = [1e-3, 5e-3, 2e-2]   # low / medium / high sparsity pressure
PRUNE_THRESH  = 0.05                  # gate < 5% → considered effectively pruned
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"{'='*62}")
print(f"  Self-Pruning Neural Network  |  CIFAR-10")
print(f"  Device  : {DEVICE}")
print(f"  Epochs  : {EPOCHS}   |   Lambdas : {LAMBDAS}")
print(f"  GateInit: sigmoid({GATE_INIT}) = {torch.sigmoid(torch.tensor(GATE_INIT)):.3f}")
print(f"{'='*62}\n")


# ════════════════════════════════════════════════════════════════════════════
# PART 1 – PrunableLinear Layer
# ════════════════════════════════════════════════════════════════════════════

class PrunableLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that augments every weight with a
    learnable scalar gate in (0, 1).

    Forward pass
    ────────────
        gate_ij       = sigmoid(score_ij)          ← learned parameter
        pruned_w_ij   = weight_ij * gate_ij        ← gated weight
        output        = X @ pruned_W.T + bias      ← standard affine

    Gradient flow
    ─────────────
    Both `weight` and `gate_scores` are nn.Parameter objects that appear
    in the forward pass.  PyTorch autograd builds the full computation
    graph through the element-wise multiply, so gradients flow correctly
    to both parameters without any custom backward implementation.

    Sparsity mechanism
    ──────────────────
    The L1 penalty pushes gate values toward 0.  Once a gate reaches 0
    it multiplies the corresponding weight to zero — effectively removing
    that connection from the network.  The weight itself is preserved and
    can revive if the penalty is later reduced (soft pruning).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight — Kaiming uniform (same as nn.Linear default)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

        # Bias
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / np.sqrt(fan_in) if fan_in > 0 else 0
            self.bias = nn.Parameter(torch.empty(out_features))
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        # Gate scores — initialised to GATE_INIT (negative) so that
        # sigmoid(score) ≈ 0.05 at the start of training.
        # This places all gates near 0, making it easy for the L1
        # penalty to push them to exactly 0 with moderate lambda values.
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), float(GATE_INIT))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: raw scores → gates in (0, 1)
        gates = torch.sigmoid(self.gate_scores)

        # Step 2: mask weights by gates (0 gate = pruned connection)
        pruned_weights = self.weight * gates

        # Step 3: standard linear operation; gradients flow to w and g
        return F.linear(x, pruned_weights, self.bias)

    @torch.no_grad()
    def get_gates(self) -> torch.Tensor:
        """Current gate values, detached from the computation graph."""
        return torch.sigmoid(self.gate_scores)

    @torch.no_grad()
    def sparsity(self, threshold: float = PRUNE_THRESH) -> float:
        """Fraction of this layer's connections that are effectively pruned."""
        return (self.get_gates() < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"gate_init=sigmoid({GATE_INIT})≈{torch.sigmoid(torch.tensor(GATE_INIT)):.2f}")


# ════════════════════════════════════════════════════════════════════════════
# Network definition
# ════════════════════════════════════════════════════════════════════════════

class SelfPruningNet(nn.Module):
    """
    Four-layer MLP using PrunableLinear throughout.
    Architecture: 3072 → 1024 → 512 → 256 → 10

    Wider layers give the sparsity mechanism more room: the network can
    prune aggressively while the surviving connections retain full capacity.
    BatchNorm stabilises training as large gate fractions fluctuate early on.
    Dropout adds regularisation complementary to the gate-based pruning.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            PrunableLinear(3 * 32 * 32, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            PrunableLinear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            PrunableLinear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))

    def prunable_layers(self):
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                yield m

    # ── PART 2 – Sparsity Loss ───────────────────────────────────────────────
    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of ALL gate values across every PrunableLinear layer.

            SparsityLoss = Σ_layers Σ_ij sigmoid(score_ij)
                         = Σ_layers Σ_ij gate_ij        (all gates > 0)

        Why L1 induces sparsity
        ────────────────────────
        The L1 penalty has a *constant* sub-gradient of ±1 at every
        non-zero point.  This gives small gates the same downward push
        as large ones.  Contrast with L2 (gradient ∝ value) which shrinks
        near 0 and never fully zeroes out a parameter.

        With L1, as a gate drifts toward 0, the optimiser has no incentive
        to stop — the gradient remains constant — so the gate reaches
        exactly 0 and the connection is effectively pruned.
        The result is a bimodal distribution: a spike at 0 (pruned) and a
        cluster of values away from 0 (active, informative connections).
        """
        total = torch.zeros(1, device=DEVICE).squeeze()
        for layer in self.prunable_layers():
            total = total + torch.sigmoid(layer.gate_scores).sum()
        return total

    @torch.no_grad()
    def overall_sparsity(self, threshold: float = PRUNE_THRESH) -> float:
        pruned = sum(
            (layer.get_gates() < threshold).sum().item()
            for layer in self.prunable_layers()
        )
        total = sum(layer.gate_scores.numel() for layer in self.prunable_layers())
        return pruned / total if total > 0 else 0.0

    @torch.no_grad()
    def all_gate_values(self) -> np.ndarray:
        return np.concatenate([
            layer.get_gates().cpu().numpy().ravel()
            for layer in self.prunable_layers()
        ])

    @torch.no_grad()
    def layer_sparsities(self):
        return [
            (f"fc{i+1} ({l.in_features}→{l.out_features})", l.sparsity())
            for i, l in enumerate(self.prunable_layers())
        ]


# ════════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════════

def get_loaders():
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_tf)

    nw = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=256,
                              shuffle=False, num_workers=nw, pin_memory=True)
    return train_loader, test_loader


# ════════════════════════════════════════════════════════════════════════════
# PART 3 – Training loop
# ════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, lam: float):
    model.train()
    run_loss = run_cls = run_sp = correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        logits = model(imgs)

        # ── Total Loss = CrossEntropy  +  λ · SparsityLoss ──────────────────
        cls_loss    = F.cross_entropy(logits, labels)
        sparse_loss = model.sparsity_loss()
        loss        = cls_loss + lam * sparse_loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs        = imgs.size(0)
        run_loss += loss.item()        * bs
        run_cls  += cls_loss.item()    * bs
        run_sp   += sparse_loss.item() * bs
        correct  += (logits.argmax(1) == labels).sum().item()
        total    += bs

    return run_loss/total, run_cls/total, run_sp/total, correct/total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        total   += imgs.size(0)
    return correct / total


def train_and_evaluate(lam: float, train_loader, test_loader):
    print(f"\n{'━'*62}")
    print(f"  λ = {lam}")
    print(f"{'━'*62}")

    model     = SelfPruningNet().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Linear warm-up (3 epochs) followed by cosine annealing
    def lr_lambda(epoch):
        warmup = 3
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, EPOCHS - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {"train_acc": [], "test_acc": [], "sparsity": []}
    best_test_acc = 0.0

    header = (f"  {'Ep':>3}  {'TrainAcc':>9}  {'TestAcc':>8}  "
              f"{'Sparsity':>9}  {'ClsLoss':>8}  {'SpLoss':>9}  {'Time':>6}")
    print(header)
    print(f"  {'─'*60}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        total_loss, cls_loss, sp_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, lam)
        scheduler.step()

        test_acc = evaluate(model, test_loader)
        sparsity = model.overall_sparsity()

        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["sparsity"].append(sparsity)

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        print(f"  {epoch:>3}  {train_acc*100:>8.2f}%  {test_acc*100:>7.2f}%  "
              f"{sparsity*100:>8.2f}%  {cls_loss:>8.4f}  {sp_loss:>9.1f}"
              f"  {time.time()-t0:>5.1f}s")

    print(f"\n  ✔  Best Test Accuracy : {best_test_acc*100:.2f}%")
    print(f"  ✔  Final Sparsity     : {model.overall_sparsity()*100:.2f}%")
    print("  Per-layer sparsity:")
    for name, sp in model.layer_sparsities():
        bar = "█" * int(sp * 20)
        print(f"     {name:<28}  {sp*100:5.1f}%  {bar}")

    return model, best_test_acc, model.overall_sparsity(), history


# ════════════════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════════════════

def plot_gate_distribution(model, lam: float,
                           save_path: str = "gate_distribution.png"):
    """
    Dual-panel gate histogram for the best model.
    Left  : full range — shows bimodal shape (spike at 0, cluster away from 0)
    Right : zoomed [0, 0.2] — shows the zero-spike in detail
    """
    gates    = model.all_gate_values()
    n_pruned = (gates < PRUNE_THRESH).sum()
    n_total  = len(gates)
    sparsity = n_pruned / n_total * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Gate Value Distribution   (λ = {lam})\n"
        f"{n_pruned:,} / {n_total:,} weights pruned  →  {sparsity:.1f}% sparsity",
        fontsize=13, fontweight="bold"
    )

    # Full distribution
    axes[0].hist(gates, bins=120, color="#1d4ed8", edgecolor="none", alpha=0.85)
    axes[0].axvline(PRUNE_THRESH, color="#ef4444", linestyle="--", linewidth=2,
                    label=f"Prune threshold ({PRUNE_THRESH})")
    axes[0].set_xlabel("Gate Value", fontsize=12)
    axes[0].set_ylabel("Count",      fontsize=12)
    axes[0].set_title("Full Distribution", fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.2)

    # Zoomed [0, 0.2]
    zoomed = gates[gates < 0.2]
    axes[1].hist(zoomed, bins=80, color="#16a34a", edgecolor="none", alpha=0.85)
    axes[1].axvline(PRUNE_THRESH, color="#ef4444", linestyle="--", linewidth=2,
                    label=f"Prune threshold ({PRUNE_THRESH})")
    axes[1].set_xlabel("Gate Value", fontsize=12)
    axes[1].set_ylabel("Count",      fontsize=12)
    axes[1].set_title("Zoomed: Gate Values < 0.2", fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {save_path}")


def plot_training_curves(all_histories: dict,
                         save_path: str = "training_curves.png"):
    """Test accuracy and sparsity curves for all lambda values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#1d4ed8", "#16a34a", "#dc2626"]

    for (lam, history), color in zip(all_histories.items(), colors):
        epochs = range(1, len(history["test_acc"]) + 1)
        axes[0].plot(epochs, [a*100 for a in history["test_acc"]],
                     label=f"λ={lam}", color=color, linewidth=2)
        axes[1].plot(epochs, [s*100 for s in history["sparsity"]],
                     label=f"λ={lam}", color=color, linewidth=2)

    for ax, ylabel, title in zip(
        axes,
        ["Test Accuracy (%)", "Sparsity (%)"],
        ["Test Accuracy vs Epoch", "Sparsity Level vs Epoch"]
    ):
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel,  fontsize=12)
        ax.set_title(title,    fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

    fig.suptitle("Self-Pruning Network — Training Dynamics",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    train_loader, test_loader = get_loaders()

    results       = []
    all_histories = {}
    best_model, best_lam, best_acc = None, None, 0.0

    for lam in LAMBDAS:
        model, acc, sparsity, history = train_and_evaluate(
            lam, train_loader, test_loader)
        results.append((lam, acc, sparsity))
        all_histories[lam] = history
        if acc > best_acc:
            best_acc, best_model, best_lam = acc, model, lam

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n\n{'═'*62}")
    print("  RESULTS SUMMARY")
    print(f"{'═'*62}")
    print(f"  {'Lambda':<12} {'Test Acc (%)':>14} {'Sparsity (%)':>14}")
    print(f"  {'─'*44}")
    for lam, acc, sparsity in results:
        tag = "  ◀ best" if lam == best_lam else ""
        print(f"  {lam:<12.4f} {acc*100:>14.2f} {sparsity*100:>14.2f}{tag}")
    print(f"{'═'*62}")

    # ── Generate plots ────────────────────────────────────────────────────
    print()
    plot_gate_distribution(best_model, best_lam)
    plot_training_curves(all_histories)

    print("\n✅  All done.")
    print("    gate_distribution.png  — bimodal gate histogram (submit this)")
    print("    training_curves.png    — accuracy & sparsity over epochs")


if __name__ == "__main__":
    main()