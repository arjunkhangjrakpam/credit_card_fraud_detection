import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import binarize
import pickle
import os
os.chdir(os.getcwd())

app = Flask(__name__)
model = pickle.load(open('credit_card_fraud_detection.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                     'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V20', 'V21', 'V22',
                     'V23', 'V25', 'V26', 'V27']
    
    df = pd.DataFrame(features_value, columns=features_name)
    #output = model.predict(df)
    output = model.predict_proba(df)
    output = binarize(output,2/10)[:,1]
        
    if output == 0:
        res_val = " Non-Fraud  "
    else:
        res_val = "** Fraud  **"
        

    return render_template('index.html', prediction_text='The Transaction is {}'.format(res_val))

if __name__ == "__main__":
    app.run()
