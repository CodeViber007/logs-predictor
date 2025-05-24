from flask import Flask, request, jsonify, render_template
from backend import predictlogs

app = Flask(__name__)

@app.route("/")
def home():
    return "<h1>LogS Predictor is Live</h1><p>Send a POST request to /predict with a SMILES string.</p>"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    smiles = data.get("smiles")
    if not smiles:
        return jsonify({"error": "No SMILES provided"}), 400

    try:
        prediction = predictlogs(smiles)
        return jsonify({"logS": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
