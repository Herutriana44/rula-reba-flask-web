from flask import Flask, jsonify, render_template, request, redirect, url_for, send_file, Response
from flask_cors import CORS
import base64
import numpy as np
import cv2
import mediapipe as mp
from models.niosh_lifting_model import calculate_niosh_lifting_equation
from models.NIOSHCalc import NIOSHCalc
from angle_calc import angle_calc
from flask_socketio import SocketIO, emit
import os
import json
import datetime
import requests

app = Flask(__name__)
# socketio = SocketIO(app)
socketio = SocketIO(app, cors_allowed_origins="*")

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

import random
import string

def generate_random_text(length=10):
    # Karakter yang akan digunakan untuk teks acak (huruf besar, huruf kecil, dan angka)
    characters = string.ascii_letters + string.digits
    # Menghasilkan teks acak dengan panjang sesuai parameter 'length'
    random_text = ''.join(random.choice(characters) for _ in range(length))
    return random_text


def eval_pose(img_path):
    frame = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    pose1=[]
    rwl = []
    li = []
    
    if results.pose_landmarks:
        # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            x_y_z=[]
            h, w,c = image_rgb.shape
            x_y_z.append(lm.x)
            x_y_z.append(lm.y)
            x_y_z.append(lm.z)
            x_y_z.append(lm.visibility)
            pose1.append(x_y_z)
            cx, cy = int(lm.x*w), int(lm.y*h)
            if id%2==0:
                cv2.circle(image_rgb, (cx, cy), 5, (255,0,0), cv2.FILLED)
            else:
                cv2.circle(image_rgb, (cx, cy), 5, (255,0,255), cv2.FILLED)

        
        # niosh_score = calculate_niosh_lifting_equation(results.pose_landmarks, frame.shape)
        # rwl.append(niosh_score['RWL'])
        # li.append(niosh_score['LI'])
        # cv2.putText(frame, f"RWL: {niosh_score['RWL']:.2f} kg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # cv2.putText(frame, f"LI: {niosh_score['LI']:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    try:
        rula,reba=angle_calc(pose1)
        rula_risk = risk_level(rula)
        reba_risk = risk_level(reba)
        res = {
            'rula': rula, 
            'reba':reba,
            'rula_risk':rula_risk,
            'reba_risk':reba_risk
            # 'rwl': rwl[-1],
            # 'li' : li[-1]
        }
        print(rula,reba)
        if (rula != "NULL") and (reba != "NULL"):
            if int(rula)>3:
                message1 = ("Rapid Upper Limb Assessment Score : "+rula+" Posture not proper in upper body. ")
                message2 = ("Posture not proper in upper body")
                msg = message1# + message2
                cv2.putText(frame, msg, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,0, 255), thickness=2, circle_radius=4))
            else:
                message1 = ("Rapid Upper Limb Assessment Score : "+rula)
                cv2.putText(frame, message1, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if int(reba)>4:
                message1 = ("Rapid Entire Body Score : "+reba+" Posture not proper in your body. ")
                message2 = ("Posture not proper in your body")
                msg = message1# + message2
                cv2.putText(frame, msg, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,0, 255), thickness=2, circle_radius=4))
            else:
                message1 = ("Rapid Entire Body Score : "+reba)
                cv2.putText(frame, message1, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # root.update()
        else:
            message1 = ("Posture Incorrect")
            cv2.putText(frame, message1, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        filename = f"{UPLOAD_FOLDER}/{generate_random_text()}.jpg"
        cv2.imwrite(filename, frame)
        return filename, res
    except Exception as e:
        print(f"error : {e}")
        filename = f"{UPLOAD_FOLDER}/{generate_random_text()}.jpg"
        cv2.imwrite(filename, frame)
        # return filename, res
        return filename, "ERROR"

camera = cv2.VideoCapture(1)
captured_image_path = 'static/captured_image.jpg'

def risk_level(num):
    try:
        num = int(num)
        if num == 1:
            return "Negligible risk = no action required"
        elif num == 2 or num == 3:
            return "Low Risk = change may be required"
        elif num >=4 and num <= 7:
            return "Medium Risk = vigilance, improvements to consider"
        elif num >= 8 and num <= 10:
            return "High Risk = Improvements needed"
        elif num > 10:
            return "Very High Risk = Immediate Response"
        else:
            return "None"
        
    except:
        return "None"
    
def niosh_risk_level(num):
    if num <= 1:
        return "The manual handling task is acceptabl"
    elif num > 1 and num <= 3:
        return "The manual handling task represents a risk of low back pain. A workstation change must be planned."
    else:
        return "The manual handling task exceeds the physical capacities of the operator. an appropriate intervention is absolutely necessary"

def gen_frames_rula():
    cap = cv2.VideoCapture(1)
    img_frame = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # img_frame.append(image_rgb)
        results = pose.process(image_rgb)
        results_hands = hands.process(image_rgb)
        img_frame.append([results, results_hands])
        
        pose1=[]
        
        if results.pose_landmarks:
            # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                x_y_z=[]
                h, w,c = image_rgb.shape
                x_y_z.append(lm.x)
                x_y_z.append(lm.y)
                x_y_z.append(lm.z)
                x_y_z.append(lm.visibility)
                pose1.append(x_y_z)
                cx, cy = int(lm.x*w), int(lm.y*h)
                if id%2==0:
                    cv2.circle(image_rgb, (cx, cy), 5, (255,0,0), cv2.FILLED)
                else:
                    cv2.circle(image_rgb, (cx, cy), 5, (255,0,255), cv2.FILLED)
        try:
            rula,reba=angle_calc(pose1)
            rula_risk = risk_level(rula)
            reba_risk = risk_level(reba)
            socketio.emit('rula_reba', {
                    'rula': rula, 
                    'reba':reba,
                    'rula_risk':rula_risk,
                    'reba_risk':reba_risk,
                })
            # print(rula,reba)
            if (rula != "NULL"):
                if int(rula)>3:
                    message1 = ("Rapid Upper Limb Assessment Score : "+rula+" Posture not proper in upper body. ")
                    message2 = ("Posture not proper in upper body")
                    msg = message1# + message2
                    cv2.putText(frame, msg, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,0, 255), thickness=2, circle_radius=4))
                else:
                    message1 = ("Rapid Upper Limb Assessment Score : "+rula)
                    cv2.putText(frame, message1, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            else:
                message1 = ("Posture Incorrect")
                cv2.putText(frame, message1, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        except Exception as e:
            print(f"error RULA REBA : {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_frames_reba():
    cap = cv2.VideoCapture(1)
    img_frame = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # img_frame.append(image_rgb)
        results = pose.process(image_rgb)
        results_hands = hands.process(image_rgb)
        img_frame.append([results, results_hands])
        
        pose1=[]
        
        if results.pose_landmarks:
            # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                x_y_z=[]
                h, w,c = image_rgb.shape
                x_y_z.append(lm.x)
                x_y_z.append(lm.y)
                x_y_z.append(lm.z)
                x_y_z.append(lm.visibility)
                pose1.append(x_y_z)
                cx, cy = int(lm.x*w), int(lm.y*h)
                if id%2==0:
                    cv2.circle(image_rgb, (cx, cy), 5, (255,0,0), cv2.FILLED)
                else:
                    cv2.circle(image_rgb, (cx, cy), 5, (255,0,255), cv2.FILLED)
        try:
            rula,reba=angle_calc(pose1)
            rula_risk = risk_level(rula)
            reba_risk = risk_level(reba)
            socketio.emit('rula_reba', {
                    'rula': rula, 
                    'reba':reba,
                    'rula_risk':rula_risk,
                    'reba_risk':reba_risk,
                })
            # print(rula,reba)
            if (reba != "NULL"):
                # # if int(rula)>3:
                # #     message1 = ("Rapid Upper Limb Assessment Score : "+rula+" Posture not proper in upper body. ")
                # #     message2 = ("Posture not proper in upper body")
                # #     msg = message1# + message2
                # #     cv2.putText(frame, msg, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # #     mp_drawing.draw_landmarks(
                # #         frame,
                # #         results.pose_landmarks,
                # #         mp_pose.POSE_CONNECTIONS,
                # #         landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,0, 255), thickness=2, circle_radius=4))
                # # else:
                #     message1 = ("Rapid Upper Limb Assessment Score : "+rula)
                #     cv2.putText(frame, message1, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                #     mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                if int(reba)>4:
                    message1 = ("Rapid Entire Body Score : "+reba+" Posture not proper in your body. ")
                    message2 = ("Posture not proper in your body")
                    msg = message1# + message2
                    cv2.putText(frame, msg, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,0, 255), thickness=2, circle_radius=4))
                else:
                    message1 = ("Rapid Entire Body Score : "+reba)
                    cv2.putText(frame, message1, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                # root.update()
            else:
                message1 = ("Posture Incorrect")
                cv2.putText(frame, message1, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        except Exception as e:
            print(f"error RULA REBA : {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_frames_niosh(bobot):
    cap = cv2.VideoCapture(1)
    img_frame = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # img_frame.append(image_rgb)
        results = pose.process(image_rgb)
        results_hands = hands.process(image_rgb)
        img_frame.append([results, results_hands])
        
        pose1=[]
        rwl = []
        li = []
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                x_y_z=[]
                h, w,c = image_rgb.shape
                x_y_z.append(lm.x)
                x_y_z.append(lm.y)
                x_y_z.append(lm.z)
                x_y_z.append(lm.visibility)
                pose1.append(x_y_z)
                cx, cy = int(lm.x*w), int(lm.y*h)
                if id%2==0:
                    cv2.circle(image_rgb, (cx, cy), 5, (255,0,0), cv2.FILLED)
                else:
                    cv2.circle(image_rgb, (cx, cy), 5, (255,0,255), cv2.FILLED)

        if len(img_frame) <= 1:
            pass
        else:
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"Time before NIOSHCalc: {current_time}")
            niosh_score = NIOSHCalc(img_frame[-2], img_frame[-1], load_weight=bobot).calculate_RWL_LI()
            try:
                rwl_ = niosh_score['RWL']
                li_ = niosh_score['LI']
                rwl.append(rwl_)
                li.append(li_)
                cv2.putText(frame, f"RWL: {niosh_score['RWL']:.2f} kg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"LI: {niosh_score['LI']:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            except Exception as e:
                print(f"error NIOSH : {e}")
            # niosh_score = calculate_niosh_lifting_equation(results.pose_landmarks, frame.shape)
            # rwl = niosh_score['RWL']
            # li = niosh_score['LI']
            # cv2.putText(frame, f"RWL: {niosh_score['RWL']:.2f} kg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # cv2.putText(frame, f"LI: {niosh_score['LI']:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        try:
            # rula,reba=angle_calc(pose1)
            # rula_risk = risk_level(rula)
            # reba_risk = risk_level(reba)
            if len(rwl) != 0 and len(li) != 0:
                socketio.emit('rula_reba', {
                    'rwl': rwl[-1],
                    'li' : li[-1],
                    'li_risk' : niosh_risk_level(li[-1])
                })
            else:
                socketio.emit('rula_reba', {
                    'rwl': 0,
                    'li' : 0,
                    'li_risk' : niosh_risk_level(0)
                })

        except Exception as e:
            print(f"error NIOSH : {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
@app.route('/video_feed_rula')
# def video_feed_rula():
#     # return Response(gen_frames_rula(), mimetype='multipart/x-mixed-replace; boundary=frame')
#     # Pastikan CORS diizinkan untuk endpoint ini
#     response = Response(gen_frames_rula(), mimetype='multipart/x-mixed-replace; boundary=frame')
#     response.headers['Access-Control-Allow-Origin'] = '*'  # Mengizinkan semua domain untuk mengakses
#     return response

@app.route('/video_feed_reba')
def video_feed_reba():
    # return Response(gen_frames_reba(), mimetype='multipart/x-mixed-replace; boundary=frame')
    # Pastikan CORS diizinkan untuk endpoint ini
    response = Response(gen_frames_reba(), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Access-Control-Allow-Origin'] = '*'  # Mengizinkan semua domain untuk mengakses
    return response


# @app.route('/upload_image', methods=['POST'])
# def upload_image():
#     data = request.json['image']
    
#     # Ambil data base64 dan hilangkan prefiks
#     header, encoded = data.split(',', 1)
#     # Decode gambar dari base64
#     img_data = base64.b64decode(encoded)
#     img_np = np.frombuffer(img_data, dtype=np.uint8)
#     frame = cv2.imdecode(img_np, flags=cv2.IMREAD_COLOR)

#     # Konversi ke RGB untuk proses
#     image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
#     # Proses pose dan deteksi tangan
#     results = pose.process(image_rgb)
#     results_hands = hands.process(image_rgb)

#     # Kirim hasil analisis melalui socket (jika diperlukan)
#     pose1 = []
#     if results.pose_landmarks:
#         for id, lm in enumerate(results.pose_landmarks.landmark):
#             x_y_z = []
#             h, w, c = image_rgb.shape
#             x_y_z.append(lm.x)
#             x_y_z.append(lm.y)
#             x_y_z.append(lm.z)
#             x_y_z.append(lm.visibility)
#             pose1.append(x_y_z)
            
#             cx, cy = int(lm.x * w), int(lm.y * h)
#             color = (255, 0, 0) if id % 2 == 0 else (255, 0, 255)
#             cv2.circle(image_rgb, (cx, cy), 5, color, cv2.FILLED)

#     # Menghitung RULA dan REBA, serta risiko
#     try:
#         rula, reba = angle_calc(pose1)
#         rula_risk = risk_level(rula)
#         reba_risk = risk_level(reba)

#         # Menambahkan hasil pada frame
#         message = f"RULA Score: {rula}"
#         if int(rula) > 3:
#             message += " - Posture not proper in upper body."
#             cv2.putText(frame, message, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         else:
#             cv2.putText(frame, message, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#     except Exception as e:
#         print(f"Error processing image: {e}")

#     # Mengembalikan gambar yang sudah diproses
#     ret, buffer = cv2.imencode('.jpg', frame)
#     processed_frame = base64.b64encode(buffer).decode('utf-8')  # Encode to base64
    
#     # Kembalikan hasil ke frontend
#     return jsonify({
#         'result': processed_frame,
#         'rula': rula,
#         'reba': reba,
#         'rula_risk': rula_risk,
#         'reba_risk': reba_risk,
#     })



@app.route('/processing_rula', methods=['POST'])
def processing_rula():
    data = request.json['image']
    
    # Ambil data base64 dan hilangkan prefiks
    header, encoded = data.split(',', 1)
    
    # Decode gambar dari base64
    img_data = base64.b64decode(encoded)
    img_np = np.frombuffer(img_data, dtype=np.uint8)
    frame = cv2.imdecode(img_np, flags=cv2.IMREAD_COLOR)

    # Konversi ke RGB untuk proses
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Proses pose dan deteksi tangan
    results = pose.process(image_rgb)

    # Untuk menyimpan dan menggambar landmark
    pose1 = []
    if results.pose_landmarks:
        h, w, c = image_rgb.shape

        # Menggambar landmark
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4))

        for id, lm in enumerate(results.pose_landmarks.landmark):
            x_y_z = [lm.x, lm.y, lm.z, lm.visibility]
            pose1.append(x_y_z)
            
            # Menggambar lingkaran di atas setiap landmark
            cx, cy = int(lm.x * w), int(lm.y * h)
            if id % 2 == 0:
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            else:
                cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Menghitung RULA dan REBA, serta risiko
        try:
            rula, reba = angle_calc(pose1)
            rula_risk = risk_level(rula)
            reba_risk = risk_level(reba)

            # Menambahkan hasil pada frame
            if rula != "NULL" and rula is not None:
                if int(rula) > 3:
                    message1 = f"RULA Score: {rula} - Posture not proper in upper body."
                    cv2.putText(frame, message1, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    message1 = f"RULA Score: {rula}"
                    cv2.putText(frame, message1, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                message1 = "Posture Incorrect"
                cv2.putText(frame, message1, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        except Exception as e:
            print(f"Error calculating RULA and REBA: {e}")
            message1 = "Error in posture calculation."
            cv2.putText(frame, message1, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Encode frame hasil ke base64
    ret, buffer = cv2.imencode('.jpg', frame)
    processed_frame = base64.b64encode(buffer).decode('utf-8')  # Encode to base64
    
    # Kembalikan hasil ke frontend
    return jsonify({
        'result': processed_frame,
        'rula': rula if 'rula' in locals() else None,
        'reba': reba if 'reba' in locals() else None,
        'rula_risk': rula_risk if 'rula_risk' in locals() else None,
        'reba_risk': reba_risk if 'reba_risk' in locals() else None,
    })

@app.route('/video_feed_rulas')
def video_feed_rula():
    return "Video feed is being processed. Please send frames via /upload_image."


@app.route('/video_feed_niosh/<int:bobot>')
def video_feed_niosh(bobot=10):
    # return Response(gen_frames_niosh(bobot), mimetype='multipart/x-mixed-replace; boundary=frame')
     # Pastikan CORS diizinkan untuk endpoint ini
    response = Response(gen_frames_niosh(bobot), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Access-Control-Allow-Origin'] = '*'  # Mengizinkan semua domain untuk mengakses
    return response

@socketio.on('connect')
def test_connect():
    print('Client connected')

@app.route('/')
def index():
    return render_template('start.html')

@app.route('/rula')
def rula_():
    return render_template('index.html', cond="rula")

@app.route('/reba')
def reba():
    return render_template('index.html', cond="reba")

@app.route('/niosh_inp')
def niosh_page():
    return render_template('niosh_inp.html')

@app.route('/niosh', methods=['GET', 'POST'])
def niosh():
    if request.method == 'POST':
        bobot = request.form['bobot']
        return render_template('index.html', cond="niosh", bobot=bobot)
    else:
        return "INPUT SALAH"


@app.route('/save_photo', methods=['GET', 'POST'])
def save_photo():
    if request.method == 'POST':
        # Mendapatkan data gambar dari request (data URL base64)
        data_url = request.form['image_data']

        # Menghapus bagian "data:image/png;base64," dari URL base64
        img_data = data_url.split(',')[1]

        # Dekode data base64
        img_data = base64.b64decode(img_data)

        # Simpan gambar ke folder
        img_filename = os.path.join(UPLOAD_FOLDER, f'captured_image.png')
        with open(img_filename, 'wb') as f:
            f.write(img_data)

        filename_res, res = eval_pose(img_filename)

        # Arahkan ke halaman yang menampilkan gambar
        return render_template('display.html', image_path=filename_res, res=res)
    
@app.route('/display_photo')
def display_photo():
    # Dapatkan path gambar dari parameter
    image_path = request.args.get('image_path', default=None)
    res = request.args.get('res', default=None)
    print("NONE" if res == None else res)

    if image_path and os.path.exists(image_path):
        # print(res)
        if res:
            try:
                res = json.loads(res)
            except json.JSONDecodeError:
                res = None  # Jika parsing gagal, set res ke None
        # Tampilkan halaman baru dengan gambar yang di-capture
        return render_template('display.html', image_path=image_path, res=res)
    else:
        return "No image found!", 404

@app.route('/capture_view')
def capture_view():
    return render_template('test.html')

@app.route('/capture')
def capture():
    # Baca frame dari kamera
    success, frame = camera.read()
    if success:
        # Implementasi evaluasi pose
        # processed_img = eval_pose(frame)

        # Simpan hasil ke file gambar
        cv2.imwrite(captured_image_path, frame)

    return redirect(url_for('index'))

@app.route('/display_capture')
def display_capture():
    # Tampilkan halaman baru dengan gambar hasil capture
    if os.path.exists(captured_image_path):
        return render_template('capture.html', image_path=captured_image_path)
    else:
        return "No image captured yet!", 404
    
    # Coba Integrasi Laravel REstFul API
@app.route('/send-to-laravel', methods=['POST'])
def send_to_laravel():
    data = {'data': 'Hello from Flask'}
    try:
        response = requests.post('http://system-health-environment.test/process-data', json=data)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")  # Menambahkan log di terminal
        return jsonify({'error': str(e)}), 500

    
# if __name__ == "__main__":
#     socketio.start_background_task(gen_frames_niosh(10))
#     socketio.start_background_task(gen_frames_rula)
#     socketio.start_background_task(gen_frames_reba)
#     CORS(app, resources={r"/*": {"origins": "*"}})  # Mengizinkan semua domain untuk akses
#     # socketio.run(app)
#     socketio.run(app, debug=True, port=5000)
#     app.run(debug=True, port=5000)

if __name__ == "__main__":
    # Start background tasks
    socketio.start_background_task(gen_frames_niosh, 10)
    socketio.start_background_task(gen_frames_rula)
    socketio.start_background_task(gen_frames_reba)
    
    # Allow all origins for CORS
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    # Run the app on the specified host and port
    socketio.run(app, debug=True, host='127.0.0.1', port=1000)
    # socketio.run(app, debug=True, port=5000)
