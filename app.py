from flask import Flask, request, render_template, jsonify, send_from_directory
from pycaret.regression import *
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
model = load_model('Final DTModel 07Sept2024')
cols = ['userId', 'gender', 'location', 'days_since_registration']

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
            
            # Read the uploaded file
            if filename.endswith('.xlsx'):
                df = pd.read_excel(filename)
            elif filename.endswith('.csv'):
                df = pd.read_csv(filename)
            
            # Ensure all required columns are present
            if not all(col in df.columns for col in cols):
                return render_template('home.html', error='Missing required columns in the uploaded file')
            
            # Make predictions
            predictions = predict_model(model, data=df)
            
            # Add predictions to the dataframe
            df['Predicted_Value'] = predictions['Label']
            
            # Save the results
            result_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'results.xlsx')
            df.to_excel(result_filename, index=False)
            
            return render_template('home.html', success='Predictions complete. Download the results file.', result_file='results.xlsx')
    
    return render_template('home.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)