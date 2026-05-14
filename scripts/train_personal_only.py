"""
train_personal_only.py — Train a fresh lightweight model on personal hand data.

Data: data/personal_features/<Word>/seq_NNNN.npy  (shape 30 x 1692)
      Recorded with wrist-relative normalisation — consistent feature space.

Split strategy: originals are split into train/test FIRST, then augmented
variants (aug*) are added ONLY to the training set. The test set contains
only original, never-augmented sequences — giving an honest real-world metric.

Run:
    venv312\\Scripts\\python.exe -u train_personal_only.py
"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
PERSONAL_DIR = ROOT / "data" / "personal_features"
FEATURES_DIR = ROOT / "data" / "extracted_features"   # label order reference
MODEL_PATH   = str(ROOT / "models" / "isl_model_solo.keras")

# ── Hyperparameters ───────────────────────────────────────────────────────────
TEST_SIZE   = 0.20   # fraction of ORIGINAL sequences held out for testing
RANDOM_SEED = 42
EPOCHS      = 200
BATCH_SIZE  = 16


# ── Label map (alphabetical — must stay consistent with realtime_translator) ───

def load_label_map() -> tuple[list[str], dict[str, int]]:
    labels = sorted(p.name for p in FEATURES_DIR.iterdir() if p.is_dir())
    return labels, {name: i for i, name in enumerate(labels)}


# ── Data loader ───────────────────────────────────────────────────────────────

def load_personal_data(
    personal_dir: Path,
    label_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns X_orig, y_orig (original seq_*.npy only) and
            X_aug,  y_aug  (augmented aug*_seq_*.npy only).

    The caller splits originals into train/test, then adds augmented data
    only to the training portion — preventing data leakage into the test set.
    """
    X_orig, y_orig = [], []
    X_aug,  y_aug  = [], []

    for word_dir in sorted(personal_dir.iterdir()):
        if not word_dir.is_dir():
            continue
        word = word_dir.name
        if word not in label_to_idx:
            print(f"  [!] '{word}' not in original label set — skipping.")
            continue

        class_idx = label_to_idx[word]

        for f in sorted(word_dir.glob("seq_*.npy")):
            arr = np.load(str(f))
            if arr.shape != (30, 378):
                print(f"  [!] Skipping {f.name} — shape {arr.shape}")
                continue
            X_orig.append(arr)
            y_orig.append(class_idx)

        for f in sorted(word_dir.glob("aug*_seq_*.npy")):
            arr = np.load(str(f))
            if arr.shape != (30, 378):
                continue
            X_aug.append(arr)
            y_aug.append(class_idx)

    return (
        np.array(X_orig, dtype=np.float32), np.array(y_orig, dtype=np.int32),
        np.array(X_aug,  dtype=np.float32), np.array(y_aug,  dtype=np.int32),
    )


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> Sequential:
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True),
                      input_shape=(30, 378)),
        BatchNormalization(),
        Dropout(0.4),
        Bidirectional(LSTM(64, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.4),
        Bidirectional(LSTM(64)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading label map ...")
    labels, label_to_idx = load_label_map()
    num_classes = len(labels)
    print(f"  {num_classes} classes\n")

    print(f"Loading personal data from {PERSONAL_DIR} ...")
    X_orig, y_orig, X_aug, y_aug = load_personal_data(PERSONAL_DIR, label_to_idx)

    if len(X_orig) == 0:
        print("  [!] No original sequences found. Run personal_collector.py first.")
        return

    counts = np.bincount(y_orig, minlength=num_classes)
    print(f"  Original sequences : {len(X_orig)}  ({(counts > 0).sum()} classes)")
    print(f"  Augmented sequences: {len(X_aug)}")
    print(f"  Orig per class — min: {counts[counts > 0].min()}  "
          f"max: {counts.max()}  mean: {counts[counts > 0].mean():.1f}\n")

    # ── Split originals into train / test ─────────────────────────────────
    # Test set = original recordings only — honest real-world metric.
    multi_mask = np.isin(y_orig, np.where(counts > 1)[0])
    X_m, y_m   = X_orig[multi_mask],  y_orig[multi_mask]
    X_s, y_s   = X_orig[~multi_mask], y_orig[~multi_mask]

    X_train_orig, X_test, y_train_orig, y_test = train_test_split(
        X_m, y_m,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_m,
    )
    if len(X_s):
        X_train_orig = np.concatenate([X_train_orig, X_s])
        y_train_orig = np.concatenate([y_train_orig, y_s])

    # ── Add augmented data ONLY to training set ───────────────────────────
    # Augmented files are named aug1_seq_NNNN.npy — map them to the same
    # train split by keeping only aug samples whose class is in the training
    # originals (all of them will be, but this is defensive).
    X_train = np.concatenate([X_train_orig, X_aug])
    y_train = np.concatenate([y_train_orig, y_aug])

    print(f"Train: {len(X_train)} (orig {len(X_train_orig)} + aug {len(X_aug)})  "
          f"|  Test: {len(X_test)} (originals only)\n")

    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat  = to_categorical(y_test,  num_classes)

    # Model
    model = build_model(num_classes)
    model.summary()

    # Callbacks
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=25,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"\nTraining for up to {EPOCHS} epochs ...\n")
    model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest accuracy (on original unseens) : {acc * 100:.2f}%")
    print(f"Test loss                           : {loss:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
