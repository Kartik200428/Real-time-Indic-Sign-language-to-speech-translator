"""
augment_personal_data.py — Expand data/personal_features/ with synthetic sequences.

For every original seq_NNNN.npy, four augmented variants are generated:
    aug1_seq_NNNN.npy  — Gaussian coordinate jitter
    aug2_seq_NNNN.npy  — Spatial scaling  (simulate closer / further camera)
    aug3_seq_NNNN.npy  — Jitter + scaling combined
    aug4_seq_NNNN.npy  — Time warp        (simulate faster / slower signing)

Only original seq_* files are augmented; running the script twice is safe
because aug* files are ignored automatically.

Run:
    venv312\\Scripts\\python.exe augment_personal_data.py
"""

from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
PERSONAL_DIR = ROOT / "data" / "personal_features"

# ── Augmentation parameters ───────────────────────────────────────────────────
NOISE_STD       = 0.008          # Gaussian jitter std  (coordinate units ≈ 0–1)
SCALE_RANGE     = (0.88, 1.12)   # uniform random scale factor
TIME_WARP_RANGE = (0.80, 1.20)   # sequence speed factor  (0.8 = faster, 1.2 = slower)
RANDOM_SEED     = 42


# ── Augmentation functions ────────────────────────────────────────────────────

def aug_noise(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add per-coordinate Gaussian noise."""
    return seq + rng.normal(0.0, NOISE_STD, seq.shape).astype(np.float32)


def aug_scale(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Uniformly scale all coordinates by a random factor.
    Simulates the signer sitting closer or further from the camera.
    """
    factor = rng.uniform(*SCALE_RANGE)
    return (seq * factor).astype(np.float32)


def aug_noise_scale(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Noise and scaling applied together."""
    return aug_scale(aug_noise(seq, rng), rng)


def aug_time_warp(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Resample the temporal axis at a random speed, then crop / pad back to
    the original length.

    speed > 1  →  the sign is performed faster  (fewer source frames used)
    speed < 1  →  the sign is performed slower   (source frames stretched out)

    After resampling, the result is always exactly (T, F) = original shape.
    """
    T, F     = seq.shape
    speed    = rng.uniform(*TIME_WARP_RANGE)
    src_len  = int(round(T * speed))
    src_len  = max(2, min(src_len, T * 2))   # clamp to a sane range

    # Build new time indices by sampling from the original T frames
    src_idx  = np.linspace(0, T - 1, src_len)
    warped   = np.zeros((src_len, F), dtype=np.float32)
    for f in range(F):
        warped[:, f] = np.interp(src_idx, np.arange(T), seq[:, f])

    # Crop to T if longer, zero-pad at the end if shorter
    if src_len >= T:
        return warped[:T]
    pad = np.zeros((T - src_len, F), dtype=np.float32)
    return np.concatenate([warped, pad], axis=0)


def aug_hand_rotate(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Apply a small 2-D in-plane rotation to the hand landmark blocks only.
    Simulates the signer's wrist being rotated slightly, which is a common
    real-world variation not covered by noise or scaling.

    Feature layout (per frame, 378 floats):
        pose  [0:132]
        face  [132:252]   40 key landmarks × 3 — nose-relative
        lh    [252:315]   21 landmarks × 3 (x,y,z) — wrist-relative
        rh    [315:378]
    """
    out  = seq.copy()
    angle = rng.uniform(-15, 15) * np.pi / 180   # -15° to +15°
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    for start in (252, 315):   # left hand, right hand
        block = out[:, start:start + 63].reshape(-1, 21, 3)  # (T, 21, 3)
        x = block[:, :, 0].copy()
        y = block[:, :, 1].copy()
        block[:, :, 0] = cos_a * x - sin_a * y
        block[:, :, 1] = sin_a * x + cos_a * y
        out[:, start:start + 63] = block.reshape(-1, 63)

    return out.astype(np.float32)


def aug_landmark_dropout(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Randomly zero out entire landmark groups for a few frames, simulating
    partial occlusion or low-confidence detections.
    """
    out       = seq.copy()
    T         = seq.shape[0]
    n_drop    = rng.integers(1, max(2, T // 5))       # drop 1 – T/5 frames
    drop_idx  = rng.choice(T, size=n_drop, replace=False)

    # For dropped frames, zero one of: left hand, right hand, or both
    for t in drop_idx:
        choice = rng.integers(0, 3)
        if choice in (0, 2):
            out[t, 252:315] = 0.0   # left hand
        if choice in (1, 2):
            out[t, 315:378] = 0.0   # right hand

    return out.astype(np.float32)


AUGMENTATIONS = [
    ("aug1", aug_noise),
    ("aug2", aug_scale),
    ("aug3", aug_noise_scale),
    ("aug4", aug_time_warp),
    ("aug5", aug_hand_rotate),
    ("aug6", aug_landmark_dropout),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    word_dirs = sorted(p for p in PERSONAL_DIR.iterdir() if p.is_dir())
    if not word_dirs:
        print(f"No data found in {PERSONAL_DIR}. Run personal_collector.py first.")
        return

    # Count originals before augmentation
    total_before = sum(
        len(list(d.glob("seq_*.npy"))) + len(list(d.glob("aug*_seq_*.npy")))
        for d in word_dirs
    )
    original_count = sum(len(list(d.glob("seq_*.npy"))) for d in word_dirs)

    print(f"Personal features directory : {PERSONAL_DIR}")
    print(f"Word classes found          : {len(word_dirs)}")
    print(f"Original sequences          : {original_count}")
    print(f"Total files before          : {total_before}")
    print(f"Augmentations per sequence  : {len(AUGMENTATIONS)}  "
          f"({', '.join(n for n, _ in AUGMENTATIONS)})\n")

    newly_created = 0

    for word_dir in word_dirs:
        orig_files = sorted(word_dir.glob("seq_*.npy"))
        if not orig_files:
            continue

        word_new = 0
        for seq_path in orig_files:
            seq = np.load(str(seq_path))            # (30, 1692)

            for tag, fn in AUGMENTATIONS:
                out_name = f"{tag}_{seq_path.name}"   # e.g. aug1_seq_0000.npy
                out_path = word_dir / out_name
                if out_path.exists():
                    continue                          # already done — safe to re-run
                augmented = fn(seq, rng)
                np.save(str(out_path), augmented)
                word_new += 1

        if word_new:
            print(f"  {word_dir.name:<25}  +{word_new} synthetic sequences")
        newly_created += word_new

    total_after = sum(
        len(list(d.glob("seq_*.npy"))) + len(list(d.glob("aug*_seq_*.npy")))
        for d in word_dirs
    )

    print(f"\n{'='*50}")
    print(f"Sequences before  : {total_before}")
    print(f"Newly created     : {newly_created}")
    print(f"Total after       : {total_after}")
    print(f"Per-class average : {total_after / len(word_dirs):.0f} sequences")
    print(f"\nNext step: retrain with  venv312\\Scripts\\python.exe -u train_personal_only.py")


if __name__ == "__main__":
    main()
