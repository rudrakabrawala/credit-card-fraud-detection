# Credit Card Fraud Detection System

## Project Overview
This project implements a machine learning-based system for detecting fraudulent credit card transactions. **Developed as part of my journey in learning Machine Learning and Data Science,** this system uses advanced techniques to identify potentially fraudulent transactions in real-time. It serves as a practical application of machine learning concepts in the financial domain, demonstrating the integration of data preprocessing, model training, and web deployment.

## Author
**Rudra Kabrawala**  
B.Tech Computer Science (3rd Year)  
NMIMS Shirpur

## Detailed Explanation
This project represents a practical exploration into the application of machine learning for fraud detection in financial transactions. The core idea is to analyze transaction data, identify patterns associated with fraudulent activities, and build a system that can detect such patterns in real-time. This process involves several key steps and technologies, providing a comprehensive learning experience.

**Key Components and Learning:**

1. **Data Preprocessing and Feature Engineering:**
   - Handling imbalanced datasets using SMOTE (Synthetic Minority Over-sampling Technique)
   - Feature scaling using StandardScaler
   - Data validation and cleaning
   - Understanding the importance of data preprocessing in real-world applications

2. **Machine Learning Model (Random Forest):**
   - Implementation of Random Forest Classifier
   - Hyperparameter tuning for optimal performance
   - Handling class imbalance in the training data
   - Model evaluation using various metrics (precision, recall, F1-score)

3. **Web Application Development (Flask):**
   - Building a user-friendly web interface
   - Real-time prediction system
   - Error handling and logging
   - Secure model deployment

4. **System Integration and Deployment:**
   - Model persistence using joblib
   - Integration of preprocessing steps with the web interface
   - Comprehensive error handling
   - Logging system for monitoring and debugging

**Challenges and Learnings:**
Developing this system presented several challenges common in real-world machine learning applications. These included handling highly imbalanced data, ensuring real-time prediction performance, implementing robust error handling, and creating a user-friendly interface. Overcoming these challenges provided hands-on experience in practical problem-solving, data handling, and deploying machine learning models in production environments.

## Features
- Real-time fraud detection for credit card transactions
- SMOTE-based handling of imbalanced dataset
- Random Forest Classifier for accurate predictions
- User-friendly web interface for transaction analysis
- Comprehensive error handling and logging
- Model persistence and easy deployment

## Technical Stack
- Python 3.7+
- scikit-learn (for machine learning algorithms)
- pandas (for data manipulation)
- Flask (for web application)
- imbalanced-learn (for SMOTE implementation)
- joblib (for model persistence)
- numpy (for numerical operations)

## Project Structure
```
credit_card_fraud_detection/
├── app.py                 # Flask web application
├── train_model.py         # Model training script
├── requirements.txt       # Project dependencies
├── templates/            # HTML templates
│   └── index.html        # Web interface template
├── fraud_model.pkl       # Trained model (generated after training)
├── scaler.pkl           # Feature scaler (generated after training)
├── creditcard.csv       # Dataset (not included in repo)
└── .gitignore          # Git ignore file
```

## Installation
To set up and run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/credit_card_fraud_detection.git
   cd credit_card_fraud_detection
   ```

2. **Create and activate virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model
To train the fraud detection model:
```bash
python train_model.py
```
This will:
- Load and preprocess the credit card transaction data
- Apply SMOTE to handle class imbalance
- Train the Random Forest model
- Save the trained model and scaler
- Display model performance metrics

### Running the Web Application
To start the web interface:
```bash
python app.py
```
The application will be available at `http://localhost:5000`

### Testing the System
You can test the system using sample transaction data. The web interface accepts 30 comma-separated numerical values representing transaction features. Here's a sample legitimate transaction:

```
0.0,-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215,149.62
```

## Model Details
- **Algorithm**: Random Forest Classifier
- **Features**: 30 anonymized features from credit card transactions
- **Class Balance**: SMOTE is used to handle imbalanced data
- **Model Parameters**:
  - n_estimators: 100
  - max_depth: 10
  - min_samples_split: 10
  - min_samples_leaf: 4
  - class_weight: 'balanced'

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Credit Card Fraud Detection Dataset
- scikit-learn for machine learning algorithms
- Flask for web application framework
- imbalanced-learn for SMOTE implementation
- The open-source community for providing valuable resources and tools

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Troubleshooting
1. If you encounter any issues with model loading, ensure that you've run `train_model.py` first
2. Make sure all dependencies are installed correctly using `pip install -r requirements.txt`
3. If the web interface doesn't start, check if port 5000 is available
4. For any data-related issues, ensure the creditcard.csv file is in the correct location 