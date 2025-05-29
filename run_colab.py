# Install required packages
!pip install flask flask-cors flask-socketio numpy opencv-python mediapipe requests python-socketio python-engineio pyngrok

# Import necessary libraries
from pyngrok import ngrok
import os
import sys
from google.colab import drive

# Mount Google Drive (optional, if you need to access files from Drive)
drive.mount('/content/drive')

# Set your ngrok authtoken
NGROK_AUTH_TOKEN = "YOUR_NGROK_AUTH_TOKEN"  # Replace with your actual ngrok auth token
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Set up ngrok tunnel
http_tunnel = ngrok.connect(5000)
print(f' * Public URL: {http_tunnel.public_url}')

# Run the Flask application
!python app.py 