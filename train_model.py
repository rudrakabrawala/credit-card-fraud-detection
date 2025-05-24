# Importing necessary libraries:
# pandas: For data loading and manipulation.
# sklearn: For machine learning tasks (model selection, scaling, model training, evaluation).
# imbalanced-learn: For handling imbalanced datasets (SMOTE).
# joblib: For saving and loading the trained model and scaler.
# numpy: For numerical operations.
# tqdm: For progress bars.
import pandas as pd
from sklearn.model_selection import train_test_split # To split data into training and testing sets.
from sklearn.ensemble import RandomForestClassifier # The classification model used.
from sklearn.preprocessing import StandardScaler # To scale features to a standard range.
from imblearn.over_sampling import SMOTE # Synthetic Minority Over-sampling Technique to balance classes.
from sklearn.metrics import classification_report # To evaluate model performance with detailed metrics.
import joblib # To save and load Python objects (model and scaler).
import numpy as np
from tqdm import tqdm

# --- Data Loading ---
# Load the credit card transaction dataset from a CSV file.
# The dataset contains transaction details and a 'Class' column indicating fraud (1) or not (0).
print("Loading dataset...")
df = pd.read_csv('creditcard.csv')

# Take a smaller sample for faster training (adjust the fraction as needed)
print("Taking a sample of the data for faster training...")
df = df.sample(frac=0.1, random_state=42)  # Using 10% of the data
print(f"Sample size: {len(df)} rows")

# Separate features (X) and target variable (y).
# 'Class' is the target variable we want to predict.
# All other columns are features used for prediction.
X = df.drop('Class', axis=1)
y = df['Class']

# --- Feature Scaling ---
# Scale the features using StandardScaler.
# This is important because some features might have very different ranges of values,
# which can affect the performance of some machine learning models.
# StandardScaler removes the mean and scales to unit variance.
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features scaled.")

# --- Handling Imbalanced Data (SMOTE) ---
# The dataset is highly imbalanced (many more legitimate transactions than fraud).
# SMOTE is used to over-sample the minority class (fraud) by creating synthetic samples.
# This helps the model learn from the minority class effectively.
print("Applying SMOTE to balance data...")
smote = SMOTE(random_state=42, sampling_strategy=0.5)  # Reduced sampling ratio
X_res, y_res = smote.fit_resample(X_scaled, y)
print(f"Data balanced. Original shape: {X_scaled.shape}, Resampled shape: {X_res.shape}")

# --- Train/Test Split ---
# Split the balanced dataset into training and testing sets.
# The training set is used to train the model, and the testing set is used to evaluate
# its performance on unseen data (20% of the data is used for testing).
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42) # random_state for reproducibility.
print("Data split into training and testing sets.")

# --- Model Training ---
# Initialize and train a RandomForestClassifier model.
# RandomForest is an ensemble method that builds multiple decision trees.
# class_weight='balanced' is used to handle potential class imbalance even after SMOTE,
# by automatically adjusting weights inversely proportional to class frequencies.
print("Training RandomForest model...")
# Optimized RandomForest parameters for faster training
model = RandomForestClassifier(
    n_estimators=100,  # Reduced number of trees
    max_depth=10,      # Limited tree depth
    min_samples_split=10,
    min_samples_leaf=4,
    n_jobs=-1,         # Use all available cores
    random_state=42,
    class_weight='balanced'
)

# Train with progress bar
model.fit(X_train, y_train)
print("Model training complete.")

# --- Model Evaluation ---
# Evaluate the trained model on the test set.
# classification_report provides precision, recall, f1-score, and support for each class.
# These metrics are crucial for evaluating models on imbalanced datasets.
print("Evaluating model...")
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred)) # Shows how well the model performed.

# --- Model and Scaler Saving ---
# Save the trained model and the scaler object to disk using joblib.
# This allows us to load the trained model and scaler later in the Flask app
# without needing to retrain the model every time.
print("Saving model and scaler...")
joblib.dump(model, 'fraud_model.pkl') # Saves the trained model.
joblib.dump(scaler, 'scaler.pkl') # Saves the scaler object.
print("Model and scaler saved successfully.") 