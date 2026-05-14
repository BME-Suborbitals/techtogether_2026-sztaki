import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_drawing = vision.drawing_utils
mp_styles = vision.drawing_styles
PoseLandmark = vision.PoseLandmark

base_options = python.BaseOptions(
    model_asset_path="pose_landmarker_full.task",
    delegate=python.BaseOptions.Delegate.CPU,
)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture("test_video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_idx = 0

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(frame_idx * 1000 / fps)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)
        frame_idx += 1

        if results.pose_landmarks:
            lm = results.pose_landmarks[0]

            mp_drawing.draw_landmarks(
                frame,
                lm,
                vision.PoseLandmarksConnections.POSE_LANDMARKS,
                mp_styles.get_default_pose_landmarks_style(),
            )

            # Access specific arm landmarks
            print("Left elbow:", lm[PoseLandmark.LEFT_ELBOW])
            print("Right wrist:", lm[PoseLandmark.RIGHT_WRIST])

        cv2.imshow("Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
