import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
from pycaret.regression import *

app = Flask(__name__)

model=load_model('final-model')

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predict',methods=['POST'])
def predict():
    int_features=[x for x in request.form.values()]
    final=np.array(int_features)
    col = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    data_unseen = pd.DataFrame([final], columns = col)
    print(int_features)
    print(final)
    prediction=predict_model(model, data=data_unseen, round = 0)
    prediction=int(prediction.prediction_label[0])
    return render_template('home.html',pred='Expected Bill will be {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    #For direct API calls throught request

    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.prediction_label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 1234, debug = True)