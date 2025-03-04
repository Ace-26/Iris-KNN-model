from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pickle
import numpy as np

# Load the trained model
with open("iris_knn_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)[0]

        flower_names = ["Setosa", "Versicolor", "Virginica"]
        predicted_flower = flower_names[prediction]

        return jsonify({"prediction": predicted_flower})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
