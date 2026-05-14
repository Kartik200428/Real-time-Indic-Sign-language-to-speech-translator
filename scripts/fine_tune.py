"""
fine_tune.py — Transfer-learning fine-tune on personal hand data.

Loads the pre-trained models/isl_model.keras, recompiles it with a very low
learning rate so the existing weights shift gently toward your hands, trains
on data/personal_features/, and saves models/isl_model_finetuned.keras.

Run:
    venv312\\Scripts\\python.exe -u fine_tune.py
"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT             = Path(__file__).parent
PERSONAL_DIR     = ROOT / "data" / "personal_features"
FEATURES_DIR     = ROOT / "data" / "extracted_features"   # used for label order only
BASE_MODEL_PATH  = str(ROOT / "models" / "isl_model.keras")
TUNED_MODEL_PATH = str(ROOT / "models" / "isl_model_finetuned.keras")

# ── Hyperparameters ───────────────────────────────────────────────────────────
FINE_TUNE_LR = 1e-4    # low enough to nudge weights without destroying them
TEST_SIZE    = 0.20
RANDOM_SEED  = 42
EPOCHS       = 50
BATCH_SIZE   = 8       # small batch suits the tiny personal dataset


# ── Labels (must match the original alphabetical order used in training) ──────

def load_label_map() -> tuple[list[str], dict[str, int]]:
    """
    Derive the 50-class label list from data/extracted_features/ — the same
    sorted() call that model_training.py used — so index 0..49 are identical.
    """
    labels = sorted(p.name for p in FEATURES_DIR.iterdir() if p.is_dir())
    label_to_idx = {name: i for i, name in enumerate(labels)}
    return labels, label_to_idx


# ── Personal data loader ──────────────────────────────────────────────────────

def load_personal_data(
    personal_dir: Path,
    label_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk personal_features/<Word>/seq_NNNN.npy and load every sequence.
    Each .npy must be shape (30, 1692).

    Returns:
        X : float32  (N, 30, 1692)
        y : int32    (N,)
    """
    sequences, targets = [], []
    missing_labels = []

    for word_dir in sorted(personal_dir.iterdir()):
        if not word_dir.is_dir():
            continue
        word = word_dir.name
        if word not in label_to_idx:
            missing_labels.append(word)
            continue

        class_idx = label_to_idx[word]
        seq_files = sorted(word_dir.glob("seq_*.npy"))
        for f in seq_files:
            arr = np.load(str(f))          # expect (30, 1692)
            if arr.shape != (30, 1692):
                print(f"  [!] Skipping {f.name} — unexpected shape {arr.shape}")
                continue
            sequences.append(arr)
            targets.append(class_idx)

    if missing_labels:
        print(f"  [!] Words in personal_features not found in original labels "
              f"(skipped): {missing_labels}")

    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets,   dtype=np.int32)
    return X, y


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # ── Labels ────────────────────────────────────────────────────────────
    print("Loading label map from original training classes ...")
    labels, label_to_idx = load_label_map()
    num_classes = len(labels)
    print(f"  {num_classes} classes  (index 0 = '{labels[0]}', "
          f"last = '{labels[-1]}')\n")

    # ── Personal dataset ──────────────────────────────────────────────────
    print(f"Loading personal data from {PERSONAL_DIR} ...")
    X, y = load_personal_data(PERSONAL_DIR, label_to_idx)

    if len(X) == 0:
        print("  [!] No personal data found. Run personal_collector.py first.")
        return

    # Per-class counts
    counts = np.bincount(y, minlength=num_classes)
    print(f"  {len(X)} total sequences across "
          f"{(counts > 0).sum()} classes")
    print(f"  Min per class: {counts[counts > 0].min()}  "
          f"Max: {counts.max()}  Mean: {counts[counts > 0].mean():.1f}\n")

    # ── Train / test split ────────────────────────────────────────────────
    # Only stratify on classes that have > 1 sample (avoids sklearn error)
    valid_mask  = np.isin(y, np.where(counts > 1)[0])
    X_v, y_v    = X[valid_mask],  y[valid_mask]
    X_s, y_s    = X[~valid_mask], y[~valid_mask]   # single-sample classes → train only

    X_train, X_test, y_train, y_test = train_test_split(
        X_v, y_v,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_v,
    )
    # Merge single-sample sequences into training set
    if len(X_s):
        X_train = np.concatenate([X_train, X_s], axis=0)
        y_train = np.concatenate([y_train, y_s], axis=0)

    print(f"Train: {len(X_train)} sequences  |  Test: {len(X_test)} sequences\n")

    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat  = to_categorical(y_test,  num_classes=num_classes)

    # ── Load base model ───────────────────────────────────────────────────
    print(f"Loading base model from {BASE_MODEL_PATH} ...")
    model = load_model(BASE_MODEL_PATH)
    print("  Base model loaded.\n")

    # ── Soft recompile (low LR — nudges weights, doesn't destroy them) ────
    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"  Recompiled with Adam lr={FINE_TUNE_LR}\n")
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=TUNED_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ── Fine-tune ─────────────────────────────────────────────────────────
    print(f"\nFine-tuning for up to {EPOCHS} epochs "
          f"(early stopping patience=10) ...\n")
    model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    # ── Final evaluation ──────────────────────────────────────────────────
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nFine-tuned test accuracy : {acc * 100:.2f}%")
    print(f"Fine-tuned test loss     : {loss:.4f}")
    print(f"Model saved to           : {TUNED_MODEL_PATH}")


if __name__ == "__main__":
    main()
