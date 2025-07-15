from flask import Flask, jsonify, request
from flask import send_from_directory
import re
import os
import numpy as np

from dashboards import create_dash_app

from application.model import model_train, model_load, model_predict
from application.model import MODEL_VERSION, MODEL_VERSION_NOTE

# Create server (flask) and Dash app
server = Flask(__name__)
app = create_dash_app(server)

@server.route('/predict', methods=['GET','POST'])
def predict():
    """
    predict function for the API
    """
    
    ## input checking
    if not request.json:
        print("ERROR: API (predict): did not receive request data")
        return jsonify([])

    if 'query' not in request.json:
        print("ERROR: API (predict): received request, but no 'query' found within")
        return jsonify([])

    ## set the test flag
    test = False
    if 'mode' in request.json and request.json['mode'] == 'test':
        test = True

    ## extract the query
    query = request.json['query']
        
    ## load model
    if test:
        data, models = model_load(prefix='test')
    else:
        data, models = model_load()
    
    if not models:
        print("ERROR: API (predict): models not available")
        return jsonify([])

    _result = model_predict(**query,all_models=models,test=test)
    result = {}
    
    ## convert numpy objects to ensure they are serializable
    for key,item in _result.items():
        if isinstance(item,np.ndarray):
            result[key] = item.tolist()
        else:
            result[key] = item
    
    return(jsonify(result))

@server.route('/train', methods=['GET','POST'])
def train():
    """
    train function for the API

    the 'mode' flag provides the ability to toggle between a test version and a 
    production verion of training
    """
    
    ## check for request data
    if not request.json:
        print("ERROR: API (train): did not receive request data")
        return jsonify(False)

    ## set the test flag
    test = False
    if 'mode' in request.json and request.json['mode'] == 'test':
        test = True

    print("... training model")
    model = model_train(test=test)
    print("... training complete")

    return(jsonify(True))

@server.route('/logs/',methods=['GET'])
def list_logs():
    """
    API endpoint to get list of logs
    """
    log_dir = os.path.join(".","logs")
    if not os.path.isdir(log_dir):
        print("ERROR: API (log): cannot find log dir")
        return jsonify([])
    
    return jsonify(os.listdir(log_dir))
        
@server.route('/logs/<filename>',methods=['GET'])
def logs(filename):
    """
    API endpoint to get logs
    """

    if not re.search(".log",filename):
        print("ERROR: API (log): file requested was not a log file: {}".format(filename))
        return jsonify([])

    log_dir = os.path.join(".","logs")
    if not os.path.isdir(log_dir):
        print("ERROR: API (log): cannot find log dir")
        return jsonify([])

    file_path = os.path.join(log_dir,filename)
    if not os.path.exists(file_path):
        print("ERROR: API (log): file requested could not be found: {}".format(filename))
        return jsonify([])
    
    return send_from_directory(log_dir, filename, as_attachment=True)

if __name__ == '__main__':
    app.run_server(host="0.0.0.0", port=8050)