## -- 1

import cv2
import mediapipe as mp
import math

# Inisialisasi Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Fungsi untuk menghitung jarak Euclidean 2D
def calculate_distance_2d(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[2] - point2[2]) ** 2)  # Gunakan sumbu x dan z

# Fungsi untuk menemukan titik tengah pergelangan kaki
def find_midpoint(point1, point2):
    mid_x = (point1[0] + point2[0]) / 2
    mid_y = (point1[1] + point2[1]) / 2
    mid_z = (point1[2] + point2[2]) / 2
    return [mid_x, mid_y, mid_z]

# Mulai kamera
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Konversi gambar ke RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Deteksi pose
        results = pose.process(image_rgb)

        # Jika ada landmark yang terdeteksi
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Landmark untuk tangan (middle knuckle)
            left_hand_knuckle = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_hand_knuckle = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

            # Landmark untuk pergelangan kaki dalam
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

            # Hitung titik tengah pergelangan kaki dalam
            mid_ankle = find_midpoint(
                [left_ankle.x, left_ankle.y, left_ankle.z], 
                [right_ankle.x, right_ankle.y, right_ankle.z]
            )

            # Gunakan tangan kiri sebagai contoh (dapat diubah sesuai kebutuhan)
            hand_position = [left_hand_knuckle.x, left_hand_knuckle.y, left_hand_knuckle.z]

            # Hitung jarak horizontal (x dan z saja)
            horizontal_distance = calculate_distance_2d(hand_position, mid_ankle)

            print(f"Jarak Horizontal (H) antara tangan dan titik tengah pergelangan kaki: {horizontal_distance:.2f}")

        # Tampilkan hasil kamera
        cv2.imshow('MediaPipe Pose', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
