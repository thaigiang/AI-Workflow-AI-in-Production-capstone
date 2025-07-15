import os,time,joblib
import numpy as np
import pandas as pd
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error
import re

from application.utils.ingestion import fetch_ts
from application.utils.logger import update_train_log,update_predict_log

MODEL_DIR = "models"
MODEL_VERSION = 1.0
MODEL_VERSION_NOTE = "FB Prophet model"

def _model_train(df,country,test=False):
    """
    example funtion to train model
    
    The 'test' flag when set to 'True':
        (1) subsets the data
        (2) specifies that the use of the 'test' log file 

    """
    ## start timer for runtime
    time_start = time.time()

    ## Getting data in FB Prophet's format
    df = df[['date', 'revenue']].copy(deep=True)
    df.rename(columns={'date':'ds', 'revenue': 'y'},inplace=True)

    if test:
        n_samples = int(np.round(0.3 * df.shape[0]))
        df = df[-n_samples:]

    ## Perform a train-test split
    n_test = int(np.round(0.2 * df.shape[0]))
    
    df_train = df[:-n_test]
    df_test = df[-n_test:]

    # Train model
    m = Prophet(weekly_seasonality=True)  
    m.fit(df_train)
    
    y_pred = m.predict(df_test)

    eval_rmse =  round(np.sqrt(mean_squared_error(df_test.y.values,y_pred.yhat)))
    
    ## retrain using all data
    m = Prophet(weekly_seasonality=True)  
    m.fit(df)

    model_name = re.sub("\.","_",str(MODEL_VERSION))
    if test:
        saved_model = os.path.join(MODEL_DIR,
                                   "test-{}-{}.joblib".format(country,model_name))
        print("... saving test version of model: {}".format(saved_model))
    else:
        saved_model = os.path.join(MODEL_DIR,
                                   "prod-{}-{}.joblib".format(country,model_name))
        print("... saving model: {}".format(saved_model))
        
    joblib.dump(m,saved_model)

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update log
    update_train_log(country,(str(df.ds.min()),str(df.ds.max())),{'rmse':eval_rmse},runtime,
                    MODEL_VERSION, MODEL_VERSION_NOTE,test=test)

def model_train(data_dir=None,test=False):
    """
    funtion to train model given a df
    """

    if not data_dir:
        data_dir = os.path.join("data","cs-train")
    
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if test:
        print("... test flag on")
        print("...... subsetting data")
        print("...... subsetting countries")
        
    ## fetch time-series formatted data
    ts_data = fetch_ts(data_dir)

    ## train a different model for each data sets
    for country,df in ts_data.items():
        
        if test and country not in ['all','united_kingdom']:
            continue
        
        _model_train(df,country,test=test)

def model_load(prefix='prod',data_dir=None):
    """
    funtion to load model
    """

    if not data_dir:
        data_dir = os.path.join("data","cs-train")
    
    models = [f for f in os.listdir(os.path.join(".",MODEL_DIR)) if re.search(prefix,f)]

    if len(models) == 0:
        raise Exception("Models with prefix '{}' cannot be found -- have these been trained?".format(prefix))

    all_models = {}
    for model in models:
        all_models[re.split("-",model)[1]] = joblib.load(os.path.join(".",MODEL_DIR,model))

    ## load data
    ts_data = fetch_ts(data_dir)
        
    return(ts_data, all_models)

def model_predict(country,year,month,day=1,n_next=None,all_models=None,test=False):
    """
    funtion to predict from model
    """

    ## start timer for runtime
    time_start = time.time()

    ## load model if needed
    if not all_models:
        if test:
            _,all_models = model_load(prefix='test')
        else:
            _,all_models = model_load()
    
    ## input checks
    if country not in all_models.keys():
        raise Exception("ERROR (model_predict) - model for country '{}' could not be found".format(country))

    for d in [year,month,day]:
        if not isinstance(d, int):
            raise Exception("ERROR (model_predict) - invalid year, month or day")
    
    ## load data
    model = all_models[country]
    #data = all_data[country]

    initial_date = "{}-{}-{}".format(year,month,day)
    ## check date
    if not n_next:
        date_range = np.arange(pd.Timestamp(initial_date).to_numpy(), (pd.Timestamp(initial_date) + pd.DateOffset(months=1)).to_numpy(), dtype='datetime64[D]')
    else:
        date_range = np.arange(pd.Timestamp(initial_date).to_numpy(), pd.Timestamp(initial_date).to_numpy() + np.timedelta64(n_next,'D'), dtype='datetime64[D]')

    df_test = pd.DataFrame({'ds':date_range})

    ## make prediction and gather data for log entry
    y_pred = model.predict(df_test)

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    query = (pd.to_datetime(date_range[0]).strftime("%Y-%m-%d"), pd.to_datetime(date_range[-1]).strftime("%Y-%m-%d"))

    ## update predict log
    update_predict_log(country,y_pred.yhat.mean(),
                        y_pred.yhat_lower.mean(), y_pred.yhat_upper.mean(),query,
                        runtime, MODEL_VERSION, test=test)
    
    return({'y_pred':y_pred.yhat.values,\
            'y_lower':y_pred.yhat_lower.values, 
            'y_upper':y_pred.yhat_upper.values,
            'date_range':date_range})

if __name__ == "__main__":

    """
    basic test procedure for model.py
    """

    ## train the model
    print("TRAINING MODELS")
    data_dir = os.path.join("data","cs-train")
    model_train(data_dir,test=True)

    ## load the model
    print("LOADING MODELS")
    all_data, all_models = model_load(prefix='test')
    print("... models loaded: ",",".join(all_models.keys()))

    ## test predict
    print("SAMPLE PREDICTION")
    country='all'
    year='2018'
    month='01'
    day='05'
    result = model_predict(country,year,month,day,test=True)
    print(result.keys())