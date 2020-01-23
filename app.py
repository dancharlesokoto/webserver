
import pickle
import numpy as np
from flask import Flask, request, jsonify

model = None
app = Flask(__name__)


def load_model():
    global model
    # model variable refers to the global variable
    with open('salesmodel.pkl', 'rb') as f:
        model = pickle.load(f)


@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        data = request.get_json()
        prediction = np.array2string(model.predict(data))

        return jsonify(prediction)


if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run()
