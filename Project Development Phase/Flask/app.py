import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas
import os
from flask import Flask, request, jsonify, render_template
from sklearn import preprocessing


app = Flask(__name__, static_url_path='/static')
model = pickle.load(open('model.pkl','rb'))
#scale = pickle.load(open('encoder.pkl','rb'))

@app.route('/')# route to display the home page
def home():
    return render_template('index.html') #rendering the home page
@app.route('/predict',methods=["POST","GET"])# route to show the show predictions in a web UI
def predict():
    # rendering the inputs given by the user
    input_feature=[float(x) for x in request.form.values()]
    features_values= [np.array(input_feature)]  
    names = [['temp','rain','snow','day','month','year','hours','minutes','seconds','weather_v2','holiday_v2']]
    data = pandas.DataFrame(features_values,columns=names)
    #predictions using the loaded model file
    prediction=model.predict(data)
    print(prediction)
    text = "The Estimated Traffic Volume is :"
    return render_template("index.html",prediction_text = text + str(prediction))
    #showing the predication results in a UI
if __name__=="__main__":
    #app.run(host='0.0.0.0', port=8000,debug=True) #runnung the app
    port=int(os.environ.get('PORT',5000))
    app.run(port=port, debug=True,use_reloader=False)