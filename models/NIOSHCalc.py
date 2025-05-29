import pandas as pd
import cv2
import mediapipe as mp
import numpy as np

class NIOSHCalc:
    def __init__(self, img1_res,img2_res, time=0.5, load_weight=10):
        # img1_pose = self.pose.process(self.img)
        # img1_hands = self.hands.process(self.img)
        # img2_pose = self.pose.process(self.dest_img)
        self.img1_pose = img1_res[0]
        self.img1_hands = img1_res[1]
        self.img2_pose = img2_res[0]
        # self.img = img
        # self.dest_img = dest_img
        self.time = time
        self.load_weight = load_weight

        # Initialize mediapipe pose and hands
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose = self.mp_pose.Pose()
        self.hands = self.mp_hands.Hands()
        # self.mp_drawing = mp.solutions.drawing_utils

        # Load B and C tables
        self.bdf = pd.read_excel("./models/Table B - NIOSH.xlsx", header=3)
        del self.bdf[self.bdf.columns.tolist()[0]]
        self.cdf = pd.read_excel("./models/Table C - NIOSH.xlsx", header=1).drop(0)
        self.cdf[self.cdf.columns.tolist()[0]] = self.cdf[self.cdf.columns.tolist()[0]].str.lower()

        # Define columns for B table
        self.d1_col = ['V < 30+', 'V ≥ 30']
        self.d2_col = ['V < 30+.1', 'V ≥ 30.1']
        self.d3_col = ['V < 30+.2', 'V ≥ 30.2']

    # Utility function to handle pre-filtering for F value
    def pre_f(self, f):
        if f <= 0.2:
            return 0.2
        elif f > 15:
            return 16
        else:
            return f

    # Access B table values
    def B_table(self, d, v, f):
        f = self.pre_f(f)
        if d <= 1:
            col = self.d1_col[0] if v < 30 else self.d1_col[1]
        elif 1 < d <= 2:
            col = self.d2_col[0] if v < 30 else self.d2_col[1]
        elif 2 < d <= 8:
            col = self.d3_col[0] if v < 30 else self.d3_col[1]
        else:
            return None
        return self.bdf[self.bdf["F"] == f][col].values[0]

    # Access C table values
    def C_table(self, c, v):
        if c != "No hands detected":
            col = self.cdf.columns.tolist()[1] if v < 75 else self.cdf.columns.tolist()[2]
            return self.cdf[self.cdf[self.cdf.columns.tolist()[0]] == c][col].values[0]
        else:
            return 0

    # Calculate midpoint between two points
    def midpoint(self, pt1, pt2):
        return [(pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2, (pt1[2] + pt2[2]) / 2]

    # Calculate H value (height)
    def H_calc(self, result):
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            mid_ankle = self.midpoint([landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y, landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].z],
                                      [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y, landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].z])

            mid_hand_knuckle = self.midpoint([landmarks[self.mp_pose.PoseLandmark.LEFT_INDEX].x, landmarks[self.mp_pose.PoseLandmark.LEFT_INDEX].y, landmarks[self.mp_pose.PoseLandmark.LEFT_INDEX].z],
                                             [landmarks[self.mp_pose.PoseLandmark.RIGHT_INDEX].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_INDEX].y, landmarks[self.mp_pose.PoseLandmark.RIGHT_INDEX].z])

            projected_hand = [mid_hand_knuckle[0], mid_ankle[1], mid_hand_knuckle[2]]
            H = abs(mid_hand_knuckle[2] - mid_ankle[1])
            return H
        return 0

    # Calculate V value (vertical height)
    def V_calc(self, result):
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            mid_hand_knuckle = self.midpoint([landmarks[self.mp_pose.PoseLandmark.LEFT_INDEX].x, landmarks[self.mp_pose.PoseLandmark.LEFT_INDEX].y, landmarks[self.mp_pose.PoseLandmark.LEFT_INDEX].z],
                                             [landmarks[self.mp_pose.PoseLandmark.RIGHT_INDEX].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_INDEX].y, landmarks[self.mp_pose.PoseLandmark.RIGHT_INDEX].z])
            V = 1 - mid_hand_knuckle[1]
            return V
        return 0

    # Calculate D value (travel distance)
    def D_calc(self, origin_result, dest_result):
        V_origin = self.V_calc(origin_result)
        V_dest = self.V_calc(dest_result)
        if V_origin is not None and V_dest is not None:
            return abs(V_dest - V_origin)
        return 0

    # Calculate A value (angle)
    def A_calc(self, result):
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            mid_ankles = self.midpoint([landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y, landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].z],
                                       [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y, landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].z])
            mid_hand_knuckle = self.midpoint([landmarks[self.mp_pose.PoseLandmark.LEFT_INDEX].x, landmarks[self.mp_pose.PoseLandmark.LEFT_INDEX].y, landmarks[self.mp_pose.PoseLandmark.LEFT_INDEX].z],
                                             [landmarks[self.mp_pose.PoseLandmark.RIGHT_INDEX].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_INDEX].y, landmarks[self.mp_pose.PoseLandmark.RIGHT_INDEX].z])
            sagittal_direction = np.array([0, 1])
            asymmetry_line = np.array([mid_hand_knuckle[0] - mid_ankles[0], mid_hand_knuckle[1] - mid_ankles[1]])
            asymmetry_line_norm = asymmetry_line / np.linalg.norm(asymmetry_line)
            dot_product = np.dot(asymmetry_line_norm, sagittal_direction)
            angle_rad = np.arccos(dot_product)
            return np.degrees(angle_rad)
        return 0

    # Grip quality classification
    def classify_grip_quality(self, pose_result, hand_result):
        if pose_result.pose_landmarks and hand_result.multi_hand_landmarks:
            landmarks = pose_result.pose_landmarks.landmark
            y_diff_hands = abs(landmarks[self.mp_pose.PoseLandmark.LEFT_INDEX].y - landmarks[self.mp_pose.PoseLandmark.RIGHT_INDEX].y)
            symmetry_threshold = 0.05
            wrist_orientation_threshold = 0.1
            y_diff_wrist = abs(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y - landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y)
            if y_diff_hands < symmetry_threshold and y_diff_wrist < wrist_orientation_threshold:
                return "good"
            elif y_diff_hands < symmetry_threshold * 2 and y_diff_wrist < wrist_orientation_threshold * 2:
                return "acceptable"
            return "bad"
        return "No hands detected"

    # Main function to calculate RWL and LI
    def calculate_RWL_LI(self):
        # img1_pose = self.pose.process(self.img)
        # img1_hands = self.hands.process(self.img)
        # img2_pose = self.pose.process(self.dest_img)

        # if self.img1_pose.pose_landmarks or self.img1_hands.multi_hand_landmarks or self.img2_pose.pose_landmarks:
        H = self.H_calc(self.img1_pose)
        V = self.V_calc(self.img1_pose)
        D = self.D_calc(self.img1_pose, self.img2_pose)
        A = self.A_calc(self.img1_pose)
        C = self.classify_grip_quality(self.img1_pose, self.img1_hands)
        F = self.time

        # NIOSH RWL formula
        LC = 51  # Load Constant
        HM = H / 5
        VM = (1 - (0.003 * (V - 75)) )
        DM = (0.82 + (4.5 / D)) / 100 if D else 1
        AM = 1 - (0.0032 * A) if A else 1
        FM = self.B_table(D, V, F)
        CM = self.C_table(C, V)

        rwl = LC * HM * VM * DM * AM * FM * CM
        print(f"{LC}, {HM}, {VM}, {DM}, {AM}, {FM}, {CM}")
        li = self.load_weight / rwl
        return {"RWL": rwl, "LI": li}
        

        # return {"RWL": 0, "LI": 0}