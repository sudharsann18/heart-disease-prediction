from flask import Flask, render_template_string, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open("heart_model.pkl", "rb"))


# Update this list to match your dataset’s features
feature_names = ['age','sex','cp','trestbps','chol','fbs','restecg',
                 'thalach','exang','oldpeak','slope','ca','thal']

# HTML page
html = '''
<!DOCTYPE html>
<html>
<head>
  <title>Heart Disease Prediction</title>
  <style>
    body { font-family: Arial; background-color: #f5f5f5; padding: 20px; }
    h1 { text-align: center; color: #d9534f; }
    form { background-color: #fff; padding: 25px; border-radius: 10px; max-width: 600px; margin: auto; }
    label { font-weight: bold; }
    input { width: 100%; padding: 8px; margin: 5px 0 15px 0; border-radius: 4px; border: 1px solid #ccc; }
    button { width: 100%; padding: 10px; background-color: #0275d8; color: white; border: none; border-radius: 5px; cursor: pointer; }
    .result { text-align: center; font-size: 20px; margin-top: 20px; }
  </style>
</head>
<body>
  <h1>❤️ Heart Disease Prediction</h1>
  <form action="/predict" method="post">
    {% for feature in feature_names %}
      <label>{{ feature }}</label>
      <input type="text" name="{{ feature }}" required><br>
    {% endfor %}
    <button type="submit">Predict</button>
  </form>
  <div class="result">
    <p>{{ prediction_text }}</p>
  </div>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(html, prediction_text='', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        vals = [float(x) for x in request.form.values()]
        arr = np.array(vals).reshape(1, -1)
        pred = model.predict(arr)[0]
        result = 'Low Risk ❤️' if pred == 0 else 'High Risk ⚠️'
        return render_template_string(html, prediction_text=f"Prediction: {result}", feature_names=feature_names)
    except Exception as e:
        return render_template_string(html, prediction_text=f"Error: {e}", feature_names=feature_names)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
