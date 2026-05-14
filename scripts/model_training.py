"""
model_training.py — Train a regularized Bidirectional LSTM model on ISL landmark sequences.

Anti-overfitting measures:
  - L2 weight decay on all LSTM and Dense kernels
  - BatchNormalization after each BiLSTM block (stabilizes activations)
  - Dropout 0.4 throughout
  - Gaussian-noise augmentation + minority-class oversampling
  - Class-weight balancing
  - ReduceLROnPlateau with patience=15 (fires late, not aggressively early)
  - EarlyStopping monitors val_accuracy with patience=30

Run:
    venv312\\Scripts\\python.exe -u model_training.py
"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization, Bidirectional, Dense, Dropout, LSTM,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
FEATURES_DIR = ROOT / "data" / "extracted_features"
MODELS_DIR = ROOT / "models"
MODEL_PATH = str(MODELS_DIR / "isl_model.keras")

# ── Hyperparameters ───────────────────────────────────────────────────────────
MAX_FRAMES = 30
FEATURE_DIM = 1692
TEST_SIZE = 0.20
RANDOM_SEED = 42
EPOCHS = 300
BATCH_SIZE = 32
MIN_SAMPLES_PER_CLASS = 45
NOISE_STD = 0.005
L2_REG = 1e-4              # weight-decay applied to all LSTM / Dense kernels


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sequences(features_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels = sorted(p.name for p in features_dir.iterdir() if p.is_dir())
    label_to_idx = {name: i for i, name in enumerate(labels)}

    sequences, targets = [], []

    for word_dir in sorted(features_dir.iterdir()):
        if not word_dir.is_dir():
            continue
        class_idx = label_to_idx[word_dir.name]

        for video_dir in sorted(word_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            frame_files = sorted(video_dir.glob("frame_*.npy"))
            if not frame_files:
                continue
            frames = np.stack([np.load(str(f)) for f in frame_files])
            frames = _pad_or_truncate(frames)
            sequences.append(frames)
            targets.append(class_idx)

    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.int32)
    return X, y, labels


def _pad_or_truncate(frames: np.ndarray) -> np.ndarray:
    T = frames.shape[0]
    if T >= MAX_FRAMES:
        return frames[:MAX_FRAMES]
    pad = np.zeros((MAX_FRAMES - T, FEATURE_DIM), dtype=frames.dtype)
    return np.concatenate([frames, pad], axis=0)


# ── Augmentation & oversampling ───────────────────────────────────────────────

def augment_sequence(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, NOISE_STD, seq.shape).astype(np.float32)
    return seq + noise


def oversample_to_minimum(
    X: np.ndarray,
    y: np.ndarray,
    min_samples: int = MIN_SAMPLES_PER_CLASS,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment minority-class training sequences with Gaussian jitter."""
    rng = np.random.default_rng(RANDOM_SEED)

    class_to_indices: dict[int, list[int]] = {}
    for idx, label in enumerate(y):
        class_to_indices.setdefault(int(label), []).append(idx)

    extra_X, extra_y = [], []
    for cls, indices in class_to_indices.items():
        deficit = min_samples - len(indices)
        if deficit <= 0:
            continue
        for _ in range(deficit):
            src = rng.choice(indices)
            extra_X.append(augment_sequence(X[src], rng))
            extra_y.append(cls)

    if not extra_X:
        return X, y

    X_out = np.concatenate([X, np.array(extra_X, dtype=np.float32)], axis=0)
    y_out = np.concatenate([y, np.array(extra_y, dtype=np.int32)], axis=0)
    return X_out, y_out


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> Sequential:
    reg = l2(L2_REG)
    model = Sequential([
        # Block 1
        Bidirectional(
            LSTM(64, return_sequences=True,
                 kernel_regularizer=reg, recurrent_regularizer=reg),
            input_shape=(MAX_FRAMES, FEATURE_DIM),
        ),
        BatchNormalization(),
        Dropout(0.4),

        # Block 2
        Bidirectional(
            LSTM(128, return_sequences=True,
                 kernel_regularizer=reg, recurrent_regularizer=reg),
        ),
        BatchNormalization(),
        Dropout(0.4),

        # Block 3
        Bidirectional(
            LSTM(64, return_sequences=False,
                 kernel_regularizer=reg, recurrent_regularizer=reg),
        ),
        BatchNormalization(),
        Dropout(0.4),

        # Classifier head
        Dense(128, activation="relu", kernel_regularizer=reg),
        Dropout(0.4),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Callbacks ─────────────────────────────────────────────────────────────────

def build_callbacks() -> list:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return [
        ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=15,       # wait longer before reducing — avoids killing LR early
            min_lr=1e-6,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=30,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading sequences from", FEATURES_DIR)
    X, y, labels = load_sequences(FEATURES_DIR)
    num_classes = len(labels)

    print(f"  {len(X)} sequences | {num_classes} classes | shape {X.shape}")
    counts = np.bincount(y)
    print(f"  Min samples/class: {counts.min()}  Max: {counts.max()}  Mean: {counts.mean():.1f}")
    print(f"  Classes: {labels}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y,
    )
    print(f"Before augmentation — Train: {len(X_train)}  Test: {len(X_test)}")

    X_train, y_train = oversample_to_minimum(X_train, y_train)
    print(f"After  augmentation — Train: {len(X_train)}  Test: {len(X_test)}\n")

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y_train,
    )
    class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}

    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat  = to_categorical(y_test,  num_classes=num_classes)

    model = build_model(num_classes)
    model.summary()

    model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=build_callbacks(),
    )

    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest accuracy : {acc * 100:.2f}%")
    print(f"Test loss     : {loss:.4f}")
    print(f"Best model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
