from flask import Flask, request, jsonify, render_template
from backend import predictlogs
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    # Check if request is JSON (from fetch)
    if request.is_json:
        data = request.get_json()
        smiles = data.get("smiles")
    else:
        # fallback to form data (if needed)
        smiles = request.form.get("smiles")

    if not smiles:
        return jsonify({"error": "No SMILES provided"}), 400

    try:
        prediction = predictlogs(smiles)
        # Ensure the prediction is JSON serializable (float or str)
        return jsonify({"logS": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
