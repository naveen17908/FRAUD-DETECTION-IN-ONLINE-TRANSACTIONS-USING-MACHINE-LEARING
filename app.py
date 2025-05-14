import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for, session, flash
import json
import os
from werkzeug.security import generate_password_hash, check_password_hash
import google.generativeai as genai
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a secure key

# Email configurations
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_FROM = 'daminmain@gmail.com'
EMAIL_PASSWORD = 'kpqtxqskedcykwjz'

# Configure Gemini-1.5-Pro API
genai.configure(api_key="AIzaSyAWuVXMmDoe-8CdczbW_Anj5pVwosnDtsA")  # Replace with your actual API key
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

# Load the saved model, scaler, and label encoders
rf_model = joblib.load('fraud_detection_rf_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = {
    'transaction_type': joblib.load('label_encoder_transaction_type.pkl'),
    'location': joblib.load('label_encoder_location.pkl'),
    'device_id': joblib.load('label_encoder_device_id.pkl'),
    'ip_address': joblib.load('label_encoder_ip_address.pkl')
}

# JSON database file
USER_DB = 'users.json'

# Initialize JSON database if it doesn't exist
if not os.path.exists(USER_DB):
    with open(USER_DB, 'w') as f:
        json.dump({}, f)

# Load users from JSON
def load_users():
    with open(USER_DB, 'r') as f:
        return json.load(f)

# Save users to JSON
def save_users(users):
    with open(USER_DB, 'w') as f:
        json.dump(users, f, indent=4)

# Send email function
def send_email(to_email, subject, body):
    try:
        # Set up the MIME
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = to_email
        msg['Subject'] = subject

        # Add body to email
        msg.attach(MIMEText(body, 'plain'))

        # Create SMTP session
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_FROM, EMAIL_PASSWORD)

        # Send email
        server.sendmail(EMAIL_FROM, to_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

# Preprocess input transaction
def preprocess_input(transaction_dict):
    df_input = pd.DataFrame([transaction_dict])
    df_input['transaction_time'] = pd.to_datetime(df_input['transaction_time'])
    df_input['hour'] = df_input['transaction_time'].dt.hour
    df_input['day_of_week'] = df_input['transaction_time'].dt.dayofweek
    df_input = df_input.drop(columns=['transaction_time'])
    
    for col, le in label_encoders.items():
        df_input[col] = df_input[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ 
                                           else le.transform([le.classes_[0]])[0])
    
    expected_cols = ['user_id', 'transaction_amount', 'transaction_type', 'location', 
                     'device_id', 'ip_address', 'is_mobile', 'hour', 'day_of_week']
    df_input = df_input[expected_cols]
    
    numerical_cols = ['transaction_amount', 'user_id', 'hour', 'day_of_week']
    df_input[numerical_cols] = scaler.transform(df_input[numerical_cols])
    
    return df_input.values

# Landing page
@app.route('/')
def landing():
    return render_template('landing.html')

# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        
        if username in users and check_password_hash(users[username]['password'], password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('predict'))
        else:
            flash('Invalid username or password', 'error')
    return render_template('login.html')

# Register page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        users = load_users()
        
        if username in users:
            flash('Username already exists', 'error')
        elif not email or '@' not in email:
            flash('Please provide a valid email address', 'error')
        else:
            users[username] = {
                'password': generate_password_hash(password),
                'email': email
            }
            save_users(users)
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

# Prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        flash('Please log in to access this page', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        transaction_dict = {
            'user_id': int(request.form['user_id']),
            'transaction_amount': float(request.form['transaction_amount']),
            'transaction_time': request.form['transaction_time'],
            'transaction_type': request.form['transaction_type'],
            'location': request.form['location'],
            'device_id': request.form['device_id'],
            'ip_address': request.form['ip_address'],
            'is_mobile': int(request.form['is_mobile'])
        }
        
        try:
            # Preprocess and predict
            processed_input = preprocess_input(transaction_dict)
            prediction = rf_model.predict(processed_input)[0]
            probability = rf_model.predict_proba(processed_input)[0, 1]
            result = "FRAUDULENT" if prediction == 1 else "NON-FRAUDULENT"
            probability_str = f"{probability:.2%}"
            
            # Prepare input for Gemini model
            gemini_prompt = (
                f"Analyze the following financial transaction and its fraud prediction result:\n"
                f"### Transaction Details:\n"
                f"- **User ID**: {transaction_dict['user_id']}\n"
                f"- **Transaction Amount**: ${transaction_dict['transaction_amount']:.2f}\n"
                f"- **Transaction Time**: {transaction_dict['transaction_time']}\n"
                f"- **Transaction Type**: {transaction_dict['transaction_type']}\n"
                f"- **Location**: {transaction_dict['location']}\n"
                f"- **Device ID**: {transaction_dict['device_id']}\n"
                f"- **IP Address**: {transaction_dict['ip_address']}\n"
                f"- **Is Mobile**: {'Yes' if transaction_dict['is_mobile'] == 1 else 'No'}\n"
                f"### Prediction Result: {result}\n"
                f"### Probability of Fraud: {probability_str}\n\n"
                f"Provide actionable suggestions to mitigate fraud risk or improve transaction security. "
                f"Consider factors such as unusual transaction patterns, location anomalies, device or IP inconsistencies, "
                f"and the predicted fraud probability. Format the suggestions as a markdown list. "
                f"For fraudulent transactions, include immediate actions (e.g., freeze account, notify user). "
                f"For non-fraudulent transactions, suggest preventive measures (e.g., enable two-factor authentication, monitor anomalies)."
            )
            
            # Call Gemini model
            try:
                gemini_response = gemini_model.generate_content([{"text": gemini_prompt}])
                ai_suggestions = gemini_response.text if gemini_response.text else "No suggestions available due to API limitations."
            except Exception as e:
                ai_suggestions = f"Error generating AI suggestions: {str(e)}"
            
            # Send email with prediction and AI suggestions
            users = load_users()
            username = session['username']
            user_email = users.get(username, {}).get('email', None)
            if user_email:
                subject = f"Fraud Detection Result for Transaction at {transaction_dict['transaction_time']}"
                body = (
                    f"Dear {username},\n\n"
                    f"A transaction was analyzed by our Fraud Detection System. Below are the details:\n\n"
                    f"### Transaction Details:\n"
                    f"- User ID: {transaction_dict['user_id']}\n"
                    f"- Amount: ${transaction_dict['transaction_amount']:.2f}\n"
                    f"- Time: {transaction_dict['transaction_time']}\n"
                    f"- Type: {transaction_dict['transaction_type']}\n"
                    f"- Location: {transaction_dict['location']}\n"
                    f"- Device ID: {transaction_dict['device_id']}\n"
                    f"- IP Address: {transaction_dict['ip_address']}\n"
                    f"- Is Mobile: {'Yes' if transaction_dict['is_mobile'] == 1 else 'No'}\n\n"
                    f"### Prediction Result: {result}\n"
                    f"### Probability of Fraud: {probability_str}\n\n"
                    f"### AI Suggestions:\n{ai_suggestions}\n\n"
                    f"Please review the suggestions and take appropriate action if necessary.\n\n"
                    f"Best regards,\nFraud Detection Team"
                )
                if not send_email(user_email, subject, body):
                    flash('Failed to send prediction email.', 'warning')
            
            # Pass result and AI suggestions to the result page
            return render_template('result.html', result=result, probability=probability_str, 
                                 transaction=transaction_dict, ai_suggestions=ai_suggestions)
        except Exception as e:
            flash(f"Error processing input: {str(e)}", 'error')
            return redirect(url_for('predict'))
    
    return render_template('predict.html')

# Result page
@app.route('/result')
def result():
    if 'username' not in session:
        flash('Please log in to access this page', 'error')
        return redirect(url_for('login'))
    return render_template('result.html')

# Logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('landing'))

# Run the app
if __name__ == '__main__':
    app.run(debug=True)