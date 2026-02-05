from flask import Flask, request, jsonify, send_from_directory
import pickle
import os


with open("model/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

THRESHOLD = -0.9

application = Flask(__name__)
app=application
@app.route("/")
def home():
    return send_from_directory('.', 'index.html')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]


    X = vectorizer.transform([text])

    score = model.decision_function(X)[0]


    label = "spam" if score > THRESHOLD else "ham"

    return jsonify({
        "text": text,
        "prediction": label,
        "score": float(score),
        "threshold": THRESHOLD
    })

if __name__ == "__main__":
    app.run(debug=True)
