import os
from pyngrok import ngrok, conf
from app import (
    app, 
    socketio, 
    gen_frames_niosh, 
    gen_frames_rula, 
    gen_frames_reba
)

def run_app():
    try:
        # Kill any existing ngrok processes
        os.system('pkill ngrok')
        
        # Kill any existing tunnels
        ngrok.kill()
        
        # Set up ngrok tunnel
        http_tunnel = ngrok.connect(5000)
        print(f' * Public URL: {http_tunnel.public_url}')
        
        # Start background tasks
        socketio.start_background_task(gen_frames_niosh, 10)
        socketio.start_background_task(gen_frames_rula)
        socketio.start_background_task(gen_frames_reba)
        
        # Run the app
        socketio.run(app, debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error running the application: {e}")
    finally:
        # Clean up ngrok when the application stops
        ngrok.kill()

if __name__ == "__main__":
    run_app()