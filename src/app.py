from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Load and split the dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'iris_model.pkl')

# Create Flask app
app = Flask(__name__)

# Load the model when the app starts
model = joblib.load('iris_model.pkl')

@app.route('/')
def home():
    return """
    <h1>Iris Prediction API</h1>
    <form action="/predict" method="POST">
        <div>
            <label>Sepal Length:</label>
            <input type="number" name="sepal_length" step="0.1" required>
        </div>
        <div>
            <label>Sepal Width:</label>
            <input type="number" name="sepal_width" step="0.1" required>
        </div>
        <div>
            <label>Petal Length:</label>
            <input type="number" name="petal_length" step="0.1" required>
        </div>
        <div>
            <label>Petal Width:</label>
            <input type="number" name="petal_width" step="0.1" required>
        </div>
        <button type="submit">Predict</button>
    </form>
    """

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Handle both form data and JSON requests
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        features = [
            float(data['sepal_length']),
            float(data['sepal_width']),
            float(data['petal_length']),
            float(data['petal_width'])
        ]
        prediction = model.predict([features])[0]
        class_name = iris.target_names[prediction]
        
        # If it's a form submission, return HTML response
        if not request.is_json:
            return f"""
            <h1>Prediction Result</h1>
            <p>The predicted iris type is: <strong>{class_name}</strong></p>
            <a href="/">Back to form</a>
            """
            
        return jsonify({'prediction': class_name})
    except Exception as e:
        return jsonify({'error': str(e)})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)