from flask import Flask, jsonify, request
import numpy as np


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict_request():  
    print(request.data)
    pass


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)