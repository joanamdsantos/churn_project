from flask import Flask, request, render_template, send_from_directory
from pycaret.classification import *
import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
model = load_model('FinalDTModel_07Sept2024')
logging.info("Transformation Pipeline and Model Successfully Loaded")

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_data(df):
    logging.info(f"Starting preprocessing. Input shape: {df.shape}")
    
    # Drop rows with missing values in specified columns
    df_valid = df.dropna(how="any", subset=["userId", "sessionId", 'gender', 'location'])
    logging.info(f"Shape after dropping NA: {df_valid.shape}")
    
    # Create flags for cancellation and downgrade events
    df_valid['cancellation_flag'] = np.where(df_valid['page'] == "Cancellation Confirmation", 1, 0)
    df_valid['downgrade_flag'] = np.where(df_valid['page'] == "Submit Downgrade", 1, 0)
    
    # Aggregate by userId to determine if they have churned via cancellation or downgrade
    user_churn = df_valid.groupby("userId").agg(
        cancellation=('cancellation_flag', 'max'),
        downgrade=('downgrade_flag', 'max')
    ).reset_index()
    
    # Create a churn column
    user_churn['churn'] = np.where(user_churn['cancellation'] == 1, 2,
                            np.where(user_churn['downgrade'] == 1, 1, 0))

    # Join the churn type back to the original data
    df_with_churn = df_valid.merge(user_churn[['userId', 'churn']], on="userId", how="left")
    logging.info(f"Shape after merging churn data: {df_with_churn.shape}")

    # Handle null values and map gender to numeric values
    df_with_churn['gender_numeric'] = df_with_churn['gender'].map({'M': 1, 'F': 0}).fillna(-1)

    # Index the location column
    df_with_churn['location_indexed'] = pd.factorize(df_with_churn['location'])[0]

    # Convert the registration timestamp to a date format
    df_with_churn['registration_date'] = pd.to_datetime(df_with_churn['registration'], unit='ms')

    # Use the number of days since registration
    df_with_churn['days_since_registration'] = (pd.Timestamp.now() - df_with_churn['registration_date']).dt.days

    logging.info(f"Preprocessing complete. Output shape: {df_with_churn.shape}")
    logging.info(f"Columns in preprocessed data: {df_with_churn.columns}")
    logging.info(f"Sample of preprocessed data:\n{df_with_churn.head()}")
    logging.info(f"Data types of columns: {df_with_churn.dtypes}")
    return df_with_churn

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('home.html', error='No file part')
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('home.html', error='No selected file')
        
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            logging.info(f"File saved: {filename}")
            
            try:
                # Read the uploaded file
                if filename.endswith('.xlsx'):
                    df = pd.read_excel(filename)
                elif filename.endswith('.csv'):
                    df = pd.read_csv(filename)
                logging.info(f"File read successfully. Shape: {df.shape}")
                
                # Preprocess the data
                df_processed = preprocess_data(df)
                
                # Ensure all required columns are present and non-empty
                required_cols = ['userId', 'gender_numeric', 'location_indexed', 'days_since_registration', 'churn']
                for col in required_cols:
                    if col not in df_processed.columns:
                        raise ValueError(f"Required column '{col}' is missing")
                    if df_processed[col].isnull().all():
                        raise ValueError(f"Required column '{col}' is empty")
                
                # Make predictions
                predictions = predict_model(model, data=df_processed)
                logging.info(f"Predictions made. Shape: {predictions.shape}")
                logging.info(f"Prediction columns: {predictions.columns}")
                
                # Add predictions to the dataframe
                if 'prediction_label' in predictions.columns:
                    df_processed['Predicted_Value'] = predictions['prediction_label']
                else:
                    raise ValueError("'prediction_label' column not found in predictions")
                
                # Save the results
                result_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv')
                df_processed.to_csv(result_filename, index=False)
                logging.info(f"Results saved to {result_filename}")
                logging.info(f"Sample of final data:\n{df_processed.head()}")
                
                return render_template('home.html', success='Predictions complete. Download the results file.', result_file='results.csv')
            except Exception as e:
                logging.error(f"Error processing file: {str(e)}")
                return render_template('home.html', error=f'Error processing file: {str(e)}')
    
    return render_template('home.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)