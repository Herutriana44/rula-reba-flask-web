from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2

app = Flask(__name__)
socketio = SocketIO(app)

camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        # ... (proses video menggunakan MediaPipe)
        # ... (mendapatkan data landmark tangan)

        # Emit data ke semua klien
        socketio.emit('new_data', {'landmark': landmark_data})

        # ... (encode frame menjadi base64 untuk ditampilkan di frontend)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def test_connect():
    print('Client connected')

if __name__ == '__main__':
    socketio.start_background_task(generate_frames)
    app.run(debug=True)