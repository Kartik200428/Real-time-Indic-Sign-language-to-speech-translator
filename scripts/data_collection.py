import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mv
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_PATH = "holistic_landmarker.task"

_FACE_CONNECTIONS = mv.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
_POSE_CONNECTIONS = mv.PoseLandmarksConnections.POSE_LANDMARKS
_HAND_CONNECTIONS = mv.HandLandmarksConnections.HAND_CONNECTIONS


# ── Model setup ──────────────────────────────────────────────────────────────

def build_holistic(model_path: str = MODEL_PATH) -> mv.HolisticLandmarker:
    options = mv.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=mv.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    return mv.HolisticLandmarker.create_from_options(options)


# ── Detection ────────────────────────────────────────────────────────────────

def detect_landmarks(
    frame: "np.ndarray",
    model: mv.HolisticLandmarker,
    timestamp_ms: int,
) -> mv.HolisticLandmarkerResult:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return model.detect_for_video(mp_image, timestamp_ms)


# ── Drawing ──────────────────────────────────────────────────────────────────

def draw_landmarks(frame: "np.ndarray", results: mv.HolisticLandmarkerResult) -> None:
    if results.face_landmarks:
        drawing_utils.draw_landmarks(
            frame,
            results.face_landmarks,
            _FACE_CONNECTIONS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
        )

    if results.pose_landmarks:
        drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            _POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
        )

    if results.left_hand_landmarks:
        drawing_utils.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            _HAND_CONNECTIONS,
            landmark_drawing_spec=drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=drawing_styles.get_default_hand_connections_style(),
        )

    if results.right_hand_landmarks:
        drawing_utils.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            _HAND_CONNECTIONS,
            landmark_drawing_spec=drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=drawing_styles.get_default_hand_connections_style(),
        )


# ── Camera helpers ───────────────────────────────────────────────────────────

def open_camera(index: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {index}.")
    return cap


# ── Main loop ────────────────────────────────────────────────────────────────

def run(camera_index: int = 0) -> None:
    cap = open_camera(camera_index)
    start_time = time.time()

    with build_holistic() as holistic:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame — skipping.")
                continue

            timestamp_ms = int((time.time() - start_time) * 1000)
            results = detect_landmarks(frame, holistic, timestamp_ms)
            draw_landmarks(frame, results)

            cv2.imshow("Sign Language — Holistic Landmarks (q to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
