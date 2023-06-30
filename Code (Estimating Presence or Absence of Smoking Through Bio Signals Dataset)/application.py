from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np

application = Flask(__name__)

model = pickle.load(open('final_model_1.pkl', 'rb'))

@application.route('/')
def hello_world():
    return render_template('home.html')

@application.route('/predict')
def home():
    return render_template('indexsp.html')

@application.route('/output', methods=['POST',"GET"])
def predict():
    features = []
    form_vals = request.form.values()
    for x in form_vals:
        if x == "Male":
            features.append(1)
        elif x == "Female":
            features.append(0)
        else:
            features.append(float(x))
    final = [np.array(features)]
    prediction = model.predict_proba(final)
    if prediction[0] == "1":
        output = "Signs of smoking present"
    else:
        output = "Signs of smoking absent"

    return render_template('smoke_prediction.html',pred=output)

if __name__ == ' __main__':
    application.run()
