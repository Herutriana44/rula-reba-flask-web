import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from models.NIOSHCalc import NIOSHCalc
from angle_calc import angle_calc

class VideoProcessor:
    def __init__(self, upload_folder):
        self.upload_folder = upload_folder
        self.allowed_extensions = {'mp4', 'avi', 'mov', 'mkv'}
        
    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def process_video_frame(self, frame, pose, hands, mp_drawing, mp_pose, analysis_type, weight=None):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        results_hands = hands.process(image_rgb)
        
        pose1 = []
        results_dict = {}
        
        if results.pose_landmarks:
            h, w, c = image_rgb.shape
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            for id, lm in enumerate(results.pose_landmarks.landmark):
                x_y_z = [lm.x, lm.y, lm.z, lm.visibility]
                pose1.append(x_y_z)
                cx, cy = int(lm.x * w), int(lm.y * h)
                color = (255, 0, 0) if id % 2 == 0 else (255, 0, 255)
                cv2.circle(frame, (cx, cy), 5, color, cv2.FILLED)
        
        try:
            if analysis_type in ['rula', 'reba']:
                rula, reba = angle_calc(pose1)
                if rula != "NULL":
                    results_dict['rula'] = rula
                if reba != "NULL":
                    results_dict['reba'] = reba
                    
            elif analysis_type == 'niosh' and weight is not None:
                niosh_score = NIOSHCalc([results, results_hands], [results, results_hands], load_weight=float(weight)).calculate_RWL_LI()
                results_dict['rwl'] = niosh_score['RWL']
                results_dict['li'] = niosh_score['LI']
                
        except Exception as e:
            print(f"Error processing frame: {e}")
        
        return frame, results_dict
    
    def process_video(self, video_file, analysis_type, weight=None, pose=None, hands=None, mp_drawing=None, mp_pose=None):
        if not self.allowed_file(video_file.filename):
            return None, "Invalid file type"
        
        # Save the uploaded video
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(self.upload_folder, filename)
        video_file.save(video_path)
        
        # Process the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Could not open video file"
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process every 5th frame to reduce processing time
        frame_interval = 5
        processed_frames = []
        all_results = []
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                processed_frame, results = self.process_video_frame(
                    frame, pose, hands, mp_drawing, mp_pose, 
                    analysis_type, weight
                )
                processed_frames.append(processed_frame)
                all_results.append(results)
            
            frame_idx += 1
        
        cap.release()
        
        # Calculate average results
        final_results = {}
        if all_results:
            for key in all_results[0].keys():
                values = [r[key] for r in all_results if key in r and r[key] is not None]
                if values:
                    if isinstance(values[0], (int, float)):
                        final_results[key] = sum(values) / len(values)
                    else:
                        # For non-numeric values, use the most common value
                        final_results[key] = max(set(values), key=values.count)
        
        # Save processed video
        output_path = os.path.join(self.upload_folder, f'processed_{filename}')
        if processed_frames:
            height, width = processed_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps/frame_interval, (width, height))
            
            for frame in processed_frames:
                out.write(frame)
            
            out.release()
            
            return {
                'results': final_results,
                'processed_video': f'/static/video_uploads/processed_{filename}'
            }, None
        
        return None, "No frames were processed" 