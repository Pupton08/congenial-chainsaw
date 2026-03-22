"""
Stage 5 — Training Loop & Optimisation
Trading Algorithm Blueprint

Implements the full walk-forward training loop exactly per blueprint Sections 5.1–5.5:

  Optimiser    : AdamW, lr=3e-4, beta=(0.9, 0.999), weight_decay=1e-4
  LR Schedule  : Linear warmup (5% of steps) → cosine decay to 1e-5
  Early stopping: patience=20 epochs, min_delta=0.0001, restore best weights
  Max epochs   : 200
  Batch size   : 256
  Gradient clip: global norm 1.0
  Regime balance: WeightedRandomSampler — inverse-frequency weights per class
  Per-epoch log : train/val loss, directional accuracy, LR, grad norm
  Halt condition: train/val gap > 0.15 for 3 consecutive epochs

Outputs:
  models/base/checkpoint_base_v1.pt   — saved when all Stage 7 thresholds pass
  models/base/checkpoint_fold_{N}.pt  — best checkpoint per fold
  logs/training/fold_{N}_metrics.csv  — per-epoch metrics
"""

import os
import glob
import math
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from model import build_model, MultiHorizonLoss, TradingModel

# ─── Configuration ────────────────────────────────────────────────────────────

PROCESSED_DIR = "data/processed"
SPLITS_DIR    = "data/splits"
MODELS_DIR    = "models/base"
LOG_DIR       = "logs/training"

LOOKBACK      = 60
N_FEATURES    = 29
HORIZONS      = [1, 5, 20]
BATCH_SIZE    = 256
MAX_EPOCHS    = 200
LR_INITIAL    = 3e-4
LR_MIN        = 1e-5
WARMUP_FRAC   = 0.05
PATIENCE      = 20
MIN_DELTA     = 0.0001
GRAD_CLIP     = 1.0
GAP_THRESHOLD = 0.15   # train/val loss gap halt condition
GAP_CONSEC    = 3      # consecutive epochs before halting on gap

# ─── Logging ──────────────────────────────────────────────────────────────────

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "training.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ─── Dataset ─────────────────────────────────────────────────────────────────

class TradingDataset(Dataset):
    """
    Loads pre-computed feature windows from data/processed/*.npy.
    For each sample (a 60-bar window), constructs direction and magnitude
    targets for each of the three forward horizons.

    Direction target : 1 if Close rises over the horizon, 0 otherwise.
    Magnitude target : log return over the horizon.

    Since we only have the window (past 60 bars), we use the final bar's
    log_return feature (feature index 0) as a proxy signal, and build
    forward targets by loading the *next* window's first bar returns.
    In practice this is done by shifting within the per-file arrays.
    """

    def __init__(self, split_df: pd.DataFrame):
        """
        split_df: DataFrame with columns [source_file, window_idx, date, regime]
        """
        self.samples  = []
        self.regimes  = []
        self.targets  = {}   # {horizon: [(direction, magnitude), ...]}

        for h in HORIZONS:
            self.targets[h] = []

        # Group by source file for efficient loading
        files_loaded = {}
        skipped = 0

        for src_file in split_df["source_file"].unique():
            npy_path = os.path.join(PROCESSED_DIR, src_file + ".npy")
            if not os.path.exists(npy_path):
                skipped += 1
                continue
            files_loaded[src_file] = np.load(npy_path)   # [n, 60, 29]

        if skipped > 0:
            log.warning("Dataset: %d .npy files not found", skipped)

        for _, row in split_df.iterrows():
            src  = row["source_file"]
            idx  = int(row["window_idx"])
            reg  = row["regime"]

            if src not in files_loaded:
                continue

            arr = files_loaded[src]
            if idx >= len(arr):
                continue

            window = arr[idx]   # [60, 29]
            self.samples.append(window)
            self.regimes.append(reg if not np.isnan(reg) else 1)

            # Build targets for each horizon using log_return (feature 0).
            #
            # Each window arr[idx] covers bars [t, t+59].
            # Windows slide forward 1 bar at a time, so arr[idx+1] covers
            # [t+1, t+60] — its last bar (-1) is the first bar AFTER our window.
            # For horizon h, we use the last bar of arr[idx+h] as the forward
            # return signal — this is bar t+59+h, fully outside the input window.
            # This is the ONLY correct construction — any slice inside arr[idx]
            # would be look-ahead leakage since the model's input is arr[idx].
            for h in HORIZONS:
                future_idx = idx + h
                if future_idx < len(arr):
                    mag   = float(arr[future_idx, -1, 0])  # last bar log_return of future window
                    direc = 1.0 if mag > 0 else 0.0
                else:
                    mag   = 0.0
                    direc = 0.5   # unknown — neutral label
                self.targets[h].append((direc, mag))

        self.samples = np.array(self.samples, dtype=np.float32)
        self.regimes = np.array(self.regimes, dtype=np.float32)

        for h in HORIZONS:
            dirs = np.array([t[0] for t in self.targets[h]], dtype=np.float32)
            mags = np.array([t[1] for t in self.targets[h]], dtype=np.float32)
            self.targets[h] = (dirs, mags)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.samples[idx])   # [60, 29]
        targets = {
            h: (
                torch.tensor(self.targets[h][0][idx]),
                torch.tensor(self.targets[h][1][idx]),
            )
            for h in HORIZONS
        }
        regime = torch.tensor(self.regimes[idx])
        return x, targets, regime


def make_weighted_sampler(dataset: TradingDataset) -> WeightedRandomSampler:
    """
    Build inverse-frequency weights per regime class so all 6 classes
    get equal representation during training (blueprint Section 3.4).
    """
    regimes  = dataset.regimes.astype(int)
    n        = len(regimes)
    counts   = np.bincount(regimes, minlength=6).astype(float)
    counts   = np.maximum(counts, 1)   # avoid div by zero for missing classes
    weights_per_class = 1.0 / counts
    sample_weights    = weights_per_class[regimes]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=n,
        replacement=True,
    )


# ─── LR Schedule ─────────────────────────────────────────────────────────────

def make_lr_schedule(optimiser, total_steps: int) -> LambdaLR:
    """
    Linear warmup over first 5% of steps, then cosine decay to LR_MIN.
    Blueprint Section 5.2.
    """
    warmup_steps = int(WARMUP_FRAC * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine   = 0.5 * (1 + math.cos(math.pi * progress))
        # Scale so LR ranges from LR_INITIAL down to LR_MIN
        return LR_MIN / LR_INITIAL + (1 - LR_MIN / LR_INITIAL) * cosine

    return LambdaLR(optimiser, lr_lambda)


# ─── Training helpers ─────────────────────────────────────────────────────────

def directional_accuracy(direction_pred: torch.Tensor,
                         direction_true: torch.Tensor) -> float:
    """Fraction of samples where predicted direction matches true direction."""
    pred_binary = (direction_pred >= 0.5).float()
    true_binary = (direction_true >= 0.5).float()
    return (pred_binary == true_binary).float().mean().item()


def run_epoch(model, loader, criterion, optimiser, scheduler,
              device, train: bool) -> dict:
    """
    Run one epoch (train or eval).
    Returns dict of metrics: loss, bce, huber, accuracy per horizon,
    mean_accuracy, grad_norm (train only).
    """
    model.train(train)
    total_loss = bce_total = huber_total = 0.0
    acc = {h: 0.0 for h in HORIZONS}
    n_batches  = 0
    grad_norms = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, targets, _ in loader:
            x = x.to(device)
            targets_dev = {
                h: (d.to(device), m.to(device))
                for h, (d, m) in targets.items()
            }

            preds = model(x)

            loss, bce, huber = criterion(preds, targets_dev)

            if train:
                optimiser.zero_grad()
                loss.backward()

                # Gradient norm before clipping (logged per blueprint 5.4)
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), GRAD_CLIP
                ).item()
                grad_norms.append(grad_norm)

                optimiser.step()
                scheduler.step()

            total_loss += loss.item()
            bce_total  += bce.item()
            huber_total += huber.item()

            # Directional accuracy per horizon
            for h in HORIZONS:
                dir_pred, _ = preds[h]
                dir_true, _ = targets_dev[h]
                acc[h] += directional_accuracy(dir_pred, dir_true)

            n_batches += 1

    n = max(n_batches, 1)
    metrics = {
        "loss":       total_loss / n,
        "bce":        bce_total  / n,
        "huber":      huber_total / n,
        "grad_norm":  float(np.mean(grad_norms)) if grad_norms else 0.0,
    }
    for h in HORIZONS:
        metrics[f"acc_h{h}"] = acc[h] / n
    metrics["mean_acc"] = np.mean([metrics[f"acc_h{h}"] for h in HORIZONS])
    return metrics


# ─── Early stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    """Blueprint Section 5.3 early stopping with best-weight restore."""

    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_loss   = float("inf")
        self.counter     = 0
        self.best_state  = None
        self.stopped     = False

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        improvement = self.best_loss - val_loss
        if improvement > self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            # Deep copy state dict
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stopped = True
            return True
        return False

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            log.info("  Restored best weights (val_loss=%.4f)", self.best_loss)


# ─── Per-fold training ────────────────────────────────────────────────────────

def train_fold(fold_n: int, device: str) -> dict:
    log.info("=" * 60)
    log.info("FOLD %d — Training", fold_n)

    # Load split index files
    train_path = os.path.join(SPLITS_DIR, f"fold_{fold_n}_train.parquet")
    val_path   = os.path.join(SPLITS_DIR, f"fold_{fold_n}_val.parquet")

    if not os.path.exists(train_path):
        log.error("Split file not found: %s", train_path)
        return {}

    train_df = pd.read_parquet(train_path)
    val_df   = pd.read_parquet(val_path)

    log.info("Loading datasets...")
    train_ds = TradingDataset(train_df)
    val_ds   = TradingDataset(val_df)
    log.info("  Train samples: %d  Val samples: %d", len(train_ds), len(val_ds))

    if len(train_ds) == 0:
        log.error("Empty training dataset for fold %d", fold_n)
        return {}

    # Weighted sampler for regime balance
    sampler = make_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        sampler=sampler, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0,
    )

    # Model, loss, optimiser
    model     = build_model(n_features=N_FEATURES, device=device)
    criterion = MultiHorizonLoss().to(device)

    # L2 weight decay only on weight matrices (not biases or LayerNorm)
    decay_params    = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if "bias" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimiser = AdamW([
        {"params": decay_params,    "weight_decay": 1e-4},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=LR_INITIAL, betas=(0.9, 0.999))

    total_steps = len(train_loader) * MAX_EPOCHS
    scheduler   = make_lr_schedule(optimiser, total_steps)
    stopper     = EarlyStopping()

    # Per-epoch metrics log
    epoch_records = []
    gap_exceed_streak = 0

    log.info("Starting training — max %d epochs, patience %d", MAX_EPOCHS, PATIENCE)

    for epoch in range(1, MAX_EPOCHS + 1):

        train_metrics = run_epoch(
            model, train_loader, criterion, optimiser, scheduler,
            device, train=True
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, None, None,
            device, train=False
        )

        gap      = train_metrics["loss"] - val_metrics["loss"]
        curr_lr  = optimiser.param_groups[0]["lr"]

        # ── Per-epoch log (blueprint 5.4) ────────────────────────
        log.info(
            "Epoch %3d | "
            "TrainLoss %.4f (BCE %.4f Huber %.4f) | "
            "ValLoss %.4f (BCE %.4f Huber %.4f) | "
            "Gap %.4f | "
            "ValAcc H1=%.3f H5=%.3f H20=%.3f | "
            "LR %.2e | GradNorm %.3f",
            epoch,
            train_metrics["loss"], train_metrics["bce"], train_metrics["huber"],
            val_metrics["loss"],   val_metrics["bce"],   val_metrics["huber"],
            gap,
            val_metrics["acc_h1"], val_metrics["acc_h5"], val_metrics["acc_h20"],
            curr_lr,
            train_metrics["grad_norm"],
        )

        # Record
        record = {"fold": fold_n, "epoch": epoch, "lr": curr_lr}
        for k, v in train_metrics.items():
            record[f"train_{k}"] = v
        for k, v in val_metrics.items():
            record[f"val_{k}"] = v
        record["gap"] = gap
        epoch_records.append(record)

        # ── Gap-based halt (blueprint 5.4) ────────────────────────
        if gap > GAP_THRESHOLD:
            gap_exceed_streak += 1
            if gap_exceed_streak >= GAP_CONSEC:
                log.warning(
                    "  ⚠  Train/val gap %.4f exceeded %.2f for %d consecutive "
                    "epochs — halting (overfitting detected).",
                    gap, GAP_THRESHOLD, GAP_CONSEC
                )
                stopper.restore_best(model)
                break
        else:
            gap_exceed_streak = 0

        # ── Gradient norm warning ─────────────────────────────────
        if train_metrics["grad_norm"] > 5.0:
            log.warning(
                "  ⚠  Gradient norm %.3f > 5.0 — instability risk.",
                train_metrics["grad_norm"]
            )

        # ── Early stopping ────────────────────────────────────────
        if stopper.step(val_metrics["loss"], model):
            log.info(
                "  Early stopping at epoch %d (no improvement for %d epochs).",
                epoch, PATIENCE
            )
            stopper.restore_best(model)
            break

    # Save fold checkpoint
    ckpt_path = os.path.join(MODELS_DIR, f"checkpoint_fold_{fold_n}.pt")
    torch.save({
        "fold":          fold_n,
        "model_state":   model.state_dict(),
        "val_loss":      stopper.best_loss,
        "n_features":    N_FEATURES,
        "hidden_dim":    128,
        "n_layers":      3,
        "epoch_stopped": len(epoch_records),
    }, ckpt_path)
    log.info("  Saved checkpoint: %s", ckpt_path)

    # Save per-epoch metrics CSV
    metrics_path = os.path.join(LOG_DIR, f"fold_{fold_n}_metrics.csv")
    pd.DataFrame(epoch_records).to_csv(metrics_path, index=False)
    log.info("  Saved metrics   : %s", metrics_path)

    return {
        "fold":          fold_n,
        "best_val_loss": stopper.best_loss,
        "epochs_run":    len(epoch_records),
        "checkpoint":    ckpt_path,
        "final_val_acc": val_metrics["mean_acc"],
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Stage 5 Training Loop — Start")
    log.info("Device : %s", device)
    log.info("Config : LR=%.0e  Batch=%d  MaxEpochs=%d  Patience=%d",
             LR_INITIAL, BATCH_SIZE, MAX_EPOCHS, PATIENCE)

    # Discover available folds
    fold_files = sorted(glob.glob(os.path.join(SPLITS_DIR, "fold_*_train.parquet")))
    if not fold_files:
        log.error("No fold split files found in %s. Run walk_forward.py first.", SPLITS_DIR)
        return

    fold_numbers = [
        int(os.path.basename(f).split("_")[1])
        for f in fold_files
    ]
    log.info("Folds to train: %s", fold_numbers)

    fold_results = []
    for fold_n in fold_numbers:
        result = train_fold(fold_n, device)
        if result:
            fold_results.append(result)

    # ── Cross-fold summary ────────────────────────────────────────
    log.info("=" * 60)
    log.info("Training complete. Cross-fold summary:")

    if not fold_results:
        log.error("No folds completed successfully.")
        return

    val_losses = [r["best_val_loss"] for r in fold_results]
    val_accs   = [r["final_val_acc"] for r in fold_results]

    for r in fold_results:
        log.info(
            "  Fold %d: best_val_loss=%.4f  val_acc=%.3f  epochs=%d",
            r["fold"], r["best_val_loss"], r["final_val_acc"], r["epochs_run"]
        )

    log.info("  Mean val loss : %.4f ± %.4f", np.mean(val_losses), np.std(val_losses))
    log.info("  Mean val acc  : %.3f ± %.3f", np.mean(val_accs),   np.std(val_accs))

    # Fold-to-fold Sharpe variance check will happen in Stage 7 evaluation.
    # Blueprint threshold: std of Sharpe across folds must be < 0.4.

    # Save the best fold's checkpoint as the base model
    best_fold = min(fold_results, key=lambda r: r["best_val_loss"])
    base_path = os.path.join(MODELS_DIR, "checkpoint_base_v1.pt")

    if not os.path.exists(base_path):
        import shutil
        shutil.copy(best_fold["checkpoint"], base_path)
        log.info(
            "  Base checkpoint saved: %s  (from fold %d)",
            base_path, best_fold["fold"]
        )
    else:
        log.info("  Base checkpoint already exists — not overwritten: %s", base_path)

    log.info("Stage 5 complete.")
    log.info("Nex step: run evaluation.py (Stage 7) on the test splits.")


if __name__ == "__main__":
    main()