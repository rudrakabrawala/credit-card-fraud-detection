# Importing necessary libraries:
# flask: The web framework to build the API.
# request, jsonify, render_template from flask: To handle web requests, return JSON responses, and render HTML templates.
# joblib: To load the trained model and scaler.
# numpy: To handle numerical data.
# logging: For better error tracking and debugging.
from flask import Flask, request, jsonify, render_template
import joblib # To load the saved model and scaler.
import numpy as np # To handle numerical operations.
import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# --- Flask App Initialization ---
# Create a Flask web application instance.
app = Flask(__name__)

# --- Model and Scaler Loading ---
def load_model_and_scaler():
    """Load the trained model and scaler with error handling."""
    try:
        if not os.path.exists('fraud_model.pkl') or not os.path.exists('scaler.pkl'):
            raise FileNotFoundError("Model files not found. Please run train_model.py first.")
        
        logger.info("Loading model and scaler...")
        model = joblib.load('fraud_model.pkl') # Load the trained RandomForest model.
        scaler = joblib.load('scaler.pkl') # Load the trained StandardScaler.
        logger.info("Model and scaler loaded successfully.")
        return model, scaler
    except Exception as e:
        logger.error(f"Error loading model files: {str(e)}")
        raise

try:
    model, scaler = load_model_and_scaler()
except Exception as e:
    logger.error(f"Failed to initialize application: {str(e)}")
    sys.exit(1)

# --- Routes ---

# Home Route:
# This route handles GET requests to the root URL ('/').
# It renders the index.html template, which contains the form for user input.
@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

# Predict Route:
# This route handles POST requests to the '/predict' URL.
# It receives transaction feature values from the HTML form, preprocesses them,
# makes a prediction using the loaded model, and returns the result.
@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests with comprehensive error handling."""
    try:
        # Get the comma-separated values from the textarea
        input_text = request.form['features']
        logger.info("Received prediction request")
        
        # Split by comma and convert to float
        try:
            values = [float(x.strip()) for x in input_text.split(',')]
        except ValueError as e:
            logger.warning(f"Invalid input format: {str(e)}")
            return render_template('index.html', 
                prediction_text='Error: Please enter valid numbers separated by commas.')
        
        # Check if we have exactly 30 features
        if len(values) != 30:
            logger.warning(f"Invalid number of features: {len(values)}")
            return render_template('index.html', 
                prediction_text='Error: Please enter exactly 30 numbers separated by commas.')
        
        # Convert to numpy array and reshape
        features = np.array([values])
        
        # Scale the features
        logger.info("Scaling input features...")
        try:
            values_scaled = scaler.transform(features)
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            return render_template('index.html', 
                prediction_text='Error: Failed to process input features.')
        
        # Make prediction
        logger.info("Making prediction...")
        try:
            prediction = model.predict(values_scaled)
            result = 'Fraud' if prediction[0] == 1 else 'Legitimate'
            logger.info(f"Prediction result: {result}")
            return render_template('index.html', prediction_text=f'Transaction is: {result}')
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return render_template('index.html', 
                prediction_text='Error: Failed to make prediction.')
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return render_template('index.html', 
            prediction_text='Error: An unexpected error occurred. Please try again.')

# --- App Execution ---
# Run the Flask application.
# debug=True allows for automatic code reloading and detailed error messages in development.
if __name__ == "__main__":
    logger.info("Starting Flask app...")
    try:
        app.run(debug=True)
    except Exception as e:
        logger.error(f"Failed to start Flask app: {str(e)}")
        sys.exit(1) 