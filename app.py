import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Load the trained model
model = joblib.load('fish_market_model.pkl')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('lab4.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Read uploaded CSV
            data = pd.read_csv(filepath)

            # Extract features (excluding Weight)
            if 'Weight' in data.columns:
                X = data.drop(columns=['Weight'])
            else:
                X = data  # If no 'Weight' column, assume all columns are features

            # Make predictions
            predictions = model.predict(X)

            # If actual weights exist, calculate metrics
            if 'Weight' in data.columns:
                y_true = data['Weight']
                mae = mean_absolute_error(y_true, predictions)
                mse = mean_squared_error(y_true, predictions)
                r2 = r2_score(y_true, predictions)
            else:
                mae, mse, r2 = None, None, None

            # Convert predictions to a list for JSON response
            return jsonify({
                "predictions": predictions.tolist(),
                "mae": round(mae, 2) if mae else None,
                "mse": round(mse, 2) if mse else None,
                "r2": round(r2, 2) if r2 else None
            })
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file format"}), 400

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
