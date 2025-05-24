from flask import Flask, request, jsonify, render_template
from backend import predictlogs
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    smiles = request.form.get("smiles")
    if not smiles:
        return jsonify({"error": "No SMILES provided"}), 400
    try:
        prediction = predictlogs(smiles)
        return render_template("index.html", prediction=prediction, smiles=smiles)
    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

