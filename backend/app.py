from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/api/ping")
def ping():
    return jsonify({"message": "Flask backend running 🎉"})

if __name__ == "__main__":
    app.run(debug=True)
