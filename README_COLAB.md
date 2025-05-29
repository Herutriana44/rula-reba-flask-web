# Running Flask Application in Google Colab

This guide will help you run the Flask application in Google Colab with ngrok integration.

## Prerequisites

1. A Google account
2. An ngrok account and auth token
3. All project files uploaded to Google Colab

## Setup Instructions

1. **Get your ngrok auth token**:
   - Sign up at https://ngrok.com/
   - Go to your dashboard
   - Copy your auth token

2. **Upload your project files to Google Colab**:
   - Create a new Colab notebook
   - Upload all project files to the Colab environment
   - Make sure to maintain the same directory structure

3. **Configure ngrok**:
   - Open `run_colab.py`
   - Replace `YOUR_NGROK_AUTH_TOKEN` with your actual ngrok auth token

4. **Run the application**:
   - In your Colab notebook, run:
   ```python
   !python run_colab.py
   ```

## Important Notes

1. The application will be accessible through the ngrok URL that is printed when you run the application
2. The ngrok URL will change each time you restart the application (unless you have a paid ngrok account)
3. Make sure all required files are in the correct directories
4. The application runs on port 5000 by default

## Troubleshooting

If you encounter any issues:

1. Check if all dependencies are installed correctly
2. Verify your ngrok auth token is correct
3. Make sure all project files are in the correct locations
4. Check the Colab console for any error messages

## File Structure

Your project should have this structure in Colab:
```
/content/
  ├── app.py
  ├── run_colab.py
  ├── angle_calc.py
  ├── models/
  │   ├── niosh_lifting_model.py
  │   └── NIOSHCalc.py
  ├── static/
  │   └── uploads/
  └── templates/
      ├── index.html
      ├── start.html
      └── other template files...
``` 