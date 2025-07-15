"""
logger tests
"""

try:
    import unittest2 as unittest
except:
    import unittest
    
import os
import csv
import unittest
from ast import literal_eval
import pandas as pd

## import model specific functions and variables
from application.utils.logger import update_train_log, update_predict_log

class LoggerTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        ensure log file is created
        """

        log_file = os.path.join("logs","train-test.log")
        if os.path.exists(log_file):
            os.remove(log_file)
        
        ## update the log
        country="brazil"
        data_range = "('2000-01-01', '2000-01-01')"
        eval_metric = {'rmse':0.5}
        runtime = "00:00:01"
        model_version = 0.1
        model_version_note = "test model"
        
        update_train_log(country,data_range,eval_metric,runtime,model_version,model_version_note,test=True)

        self.assertTrue(os.path.exists(log_file))
        
    def test_02_train(self):
        """
        ensure that content can be retrieved from log file
        """

        log_file = os.path.join("logs","train-test.log")
        
        ## update the log
        country="brazil"
        data_range = "('2000-01-01', '2000-01-01')"
        eval_metric = {'rmse':0.5}
        runtime = "00:00:01"
        model_version = 0.1
        model_version_note = "test model"
        
        update_train_log(country,data_range,eval_metric,runtime,model_version,model_version_note,test=True)

        df = pd.read_csv(log_file)
        logged_eval_metric = [literal_eval(i) for i in df['eval_metric'].copy()][-1]
        self.assertEqual(eval_metric,logged_eval_metric)
                

    def test_03_predict(self):
        """
        ensure log file is created
        """

        log_file = os.path.join("logs","predict-test.log")
        if os.path.exists(log_file):
            os.remove(log_file)
        
        ## update the log
        country = "brazil"
        y_pred = 0
        y_lower = -1
        y_upper = 1
        query = "('2000-01-01', '2000-01-01')"
        runtime = "00:00:02"
        model_version = 0.1

        update_predict_log(country,y_pred, y_lower, y_upper,\
                      query, runtime, model_version, test=True)
        self.assertTrue(os.path.exists(log_file))

    
    def test_04_predict(self):
        """
        ensure that content can be retrieved from log file
        """

        log_file = os.path.join("logs","predict-test.log")

        ## update the log
        country = "brazil"
        y_pred = 0
        y_lower = -1
        y_upper = 1
        query = "('2000-01-01', '2000-01-01')"
        runtime = "00:00:02"
        model_version = 0.1

        update_predict_log(country,y_pred, y_lower, y_upper,\
                      query, runtime, model_version, test=True)

        df = pd.read_csv(log_file)

        logged_y_pred = [int(i) for i in df['y_pred'].copy()][-1]
        self.assertEqual(y_pred,logged_y_pred)


### Run the tests
if __name__ == '__main__':
    unittest.main()
      
