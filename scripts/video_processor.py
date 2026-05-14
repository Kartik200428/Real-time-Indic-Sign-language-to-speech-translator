"""
video_processor.py — Extract MediaPipe holistic landmarks from INCLUDE dataset videos.

Output: data/extracted_features/<Word>/<VideoStem>/frame_NNNN.npy
Array shape per frame: (1692,)
  = pose(132) + face(1434) + left_hand(63) + right_hand(63)
  Pose   : 33 landmarks × 4 values (x, y, z, visibility)
  Face   : 478 landmarks × 3 values (x, y, z)
  Hands  : 21 landmarks × 3 values (x, y, z) each

NOTE: mediapipe 0.10+ removed solutions; this script uses the Tasks API.
"""

import re
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mv
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "extracted_features"
MODEL_PATH = str(ROOT / "holistic_landmarker.task")

# ── Landmark dimension constants ──────────────────────────────────────────────
POSE_DIM = 33 * 4    # 132
FACE_DIM = 478 * 3   # 1434  (Tasks API holistic uses 478-point face mesh)
HAND_DIM = 21 * 3    # 63
KEYPOINT_DIM = POSE_DIM + FACE_DIM + HAND_DIM * 2  # 1692

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

# ── 50 target words ───────────────────────────────────────────────────────────
TARGET_WORDS = [
    # Pronouns
    "I", "you", "he", "she", "they", "we", "it",
    # People / Jobs
    "Mother", "Father", "Brother", "Sister", "Friend", "Family", "Man", "Woman",
    "Teacher", "Doctor", "Police", "Student",
    # Verbs / Actions (limited in dataset)
    "Exercise", "Sign", "Dream", "Sport",
    # States / Adjectives
    "good", "bad", "happy", "sad", "sick", "healthy",
    "strong", "weak", "young", "old", "Deaf", "alive",
    # Time
    "Today", "Tomorrow", "Yesterday", "Morning", "Night", "Time",
    # Questions / Greetings
    "How are you", "Hello", "Thank you", "Good Morning",
    # Common Nouns / Places
    "Hospital", "School", "House", "Medicine", "Patient",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    """Fold to lowercase, strip spaces/underscores/hyphens for fuzzy matching."""
    return re.sub(r"[\s_\-]", "", s).lower()


def _word_from_folder(folder_name: str) -> str:
    """'40. I' → 'I',  '55. Thank you' → 'Thank you'"""
    m = re.match(r"^\d+\.\s*(.+)$", folder_name)
    return m.group(1).strip() if m else folder_name.strip()


def scan_word_folders(raw_dir: Path) -> dict[str, tuple[str, Path]]:
    """Return {normalized_word: (display_name, path)} for every leaf word folder."""
    folders: dict[str, tuple[str, Path]] = {}
    for category in raw_dir.iterdir():
        if not category.is_dir():
            continue
        for word_dir in category.iterdir():
            if not word_dir.is_dir():
                continue
            display = _word_from_folder(word_dir.name)
            key = _normalize(display)
            existing = folders.get(key)
            if existing is None:
                folders[key] = (display, word_dir)
            else:
                # keep the folder with more videos on collision
                if _count_videos(word_dir) > _count_videos(existing[1]):
                    folders[key] = (display, word_dir)
    return folders


def _count_videos(folder: Path) -> int:
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS)


def select_folders(
    all_folders: dict[str, tuple[str, Path]],
    target_words: list[str],
    n: int = 50,
) -> list[tuple[str, Path]]:
    """
    Match TARGET_WORDS first (using normalised spelling as the output label).
    If fewer than n match, fall back to the top-n folders by video count.
    """
    matched: dict[str, tuple[str, Path]] = {}
    for word in target_words:
        key = _normalize(word)
        if key in all_folders:
            _, path = all_folders[key]
            matched[word] = (word, path)  # use TARGET_WORDS spelling as label

    if len(matched) >= n:
        print(f"Matched all {n} target words in the dataset.")
        return list(matched.values())[:n]

    print(
        f"Matched {len(matched)}/{n} target words. "
        f"Falling back to top-{n} folders by video count."
    )
    ranked = sorted(
        all_folders.values(),
        key=lambda t: _count_videos(t[1]),
        reverse=True,
    )
    return [(display, path) for display, path in ranked[:n]]


# ── MediaPipe setup ───────────────────────────────────────────────────────────

def build_holistic() -> mv.HolisticLandmarker:
    options = mv.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mv.RunningMode.IMAGE,  # IMAGE mode: each frame is independent
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    return mv.HolisticLandmarker.create_from_options(options)


def extract_keypoints(result: mv.HolisticLandmarkerResult) -> np.ndarray:
    """Flatten all landmarks into a fixed-length (KEYPOINT_DIM,) array."""
    # HolisticLandmarkerResult gives flat lists directly (not list-of-lists)
    if result.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks]
        ).flatten()
    else:
        pose = np.zeros(POSE_DIM)

    if result.face_landmarks:
        face = np.array(
            [[lm.x, lm.y, lm.z] for lm in result.face_landmarks]
        ).flatten()
        # Guard against model returning a different landmark count
        if face.size < FACE_DIM:
            face = np.concatenate([face, np.zeros(FACE_DIM - face.size)])
        else:
            face = face[:FACE_DIM]
    else:
        face = np.zeros(FACE_DIM)

    if result.left_hand_landmarks:
        lh = np.array(
            [[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks]
        ).flatten()
    else:
        lh = np.zeros(HAND_DIM)

    if result.right_hand_landmarks:
        rh = np.array(
            [[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks]
        ).flatten()
    else:
        rh = np.zeros(HAND_DIM)

    return np.concatenate([pose, face, lh, rh])


# ── Core processing ───────────────────────────────────────────────────────────

def process_video(
    video_path: Path,
    label: str,
    holistic: mv.HolisticLandmarker,
    out_dir: Path,
) -> int:
    """Extract keypoints for every frame of one video. Returns frames saved."""
    dest = out_dir / label / video_path.stem
    dest.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = holistic.detect(mp_image)
            keypoints = extract_keypoints(result)
            np.save(str(dest / f"frame_{frame_idx:04d}.npy"), keypoints)
            frame_idx += 1
    finally:
        cap.release()

    return frame_idx


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="ISL landmark extractor")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the 50 selected words and video counts, then exit."
    )
    args = parser.parse_args()

    all_folders = scan_word_folders(RAW_DIR)
    selected = select_folders(all_folders, TARGET_WORDS, n=50)

    all_videos: list[tuple[str, Path]] = []
    for label, folder in selected:
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
                all_videos.append((label, f))

    if args.dry_run:
        print("\n--- Dry run: selected words ---")
        video_counts = {}
        for label, video_path in all_videos:
            video_counts[label] = video_counts.get(label, 0) + 1
        for i, (label, folder) in enumerate(selected, 1):
            count = video_counts.get(label, 0)
            source = folder.parent.name  # category folder name
            print(f"  {i:>2}. {label:<30}  {count:>2} videos  [{source}]")
        print(f"\nTotal: {len(selected)} words, {len(all_videos)} videos")
        return

    print(f"\nProcessing {len(all_videos)} videos across {len(selected)} words.")
    print(f"Output  ->  {OUT_DIR}")
    print(f"Feature vector size: {KEYPOINT_DIM} values per frame\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    skipped = 0
    total_frames = 0

    with build_holistic() as holistic:
        for label, video_path in tqdm(all_videos, desc="Videos", unit="vid"):
            try:
                frames = process_video(video_path, label, holistic, OUT_DIR)
                total_frames += frames
            except Exception as exc:
                tqdm.write(f"  [SKIP] {video_path.name} ({label}) — {exc}")
                skipped += 1

    print(f"\nDone.  {total_frames:,} frames saved | {skipped} videos skipped.")


if __name__ == "__main__":
    main()
