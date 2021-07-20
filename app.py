import pandas as pd
from flask import Flask, jsonify, request
import pickle

import warnings
warnings.filterwarnings('ignore')
from Model import *


# load model
if __name__=='__main__':
    # with open('model12.pkl', 'rb') as f:
    #     mod = pickle.load(f)
    mod = pickle.load(open('modelnew.pkl', 'rb'))


# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():
    # get data
    data = request.get_json(force=True)

    # convert data into list


    # predictions
    
    result = mod.predict(data)

    # send back to browser
    output = {'results': result}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
