"""
model tests
"""

try:
    import unittest2 as unittest
except:
    import unittest

import os
import csv
import numbers
from ast import literal_eval
import pandas as pd

## import model specific functions and variables
from application.model import *

class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        test the train functionality
        """
        #data_dir = os.path.join('data', 'cs-train')

        ## train the model
        model_train(test=True)

        # assert model folder exists and contains models (files ending with '.joblib')
        self.assertTrue(os.path.exists(MODEL_DIR))
        self.assertTrue(len([file for file in os.listdir(MODEL_DIR) if file.endswith(".joblib")])>0)

    def test_02_load(self):
        """
        test the train functionality
        """
                        
        ## train the model
        _, all_models = model_load(prefix='test')
        
        for country, model in all_models.items():
            self.assertTrue('predict' in dir(model))
            self.assertTrue('fit' in dir(model))

       
    def test_03_predict(self):
        """
        test the predict function input
        """
    
        ## query to be passed
        country = 'all'
        year = 2018
        month = 2
        n_next = 20

        result = model_predict(country,year,month,n_next=n_next,test=True)
        y_pred = result['y_pred']
        y_lower = result['y_lower']
        y_upper = result['y_upper']

        self.assertTrue(len(y_pred)==n_next)
        self.assertTrue(len(y_lower)==n_next)
        self.assertTrue(len(y_upper)==n_next)

        for i in y_pred:
            self.assertTrue(isinstance(i, numbers.Number))

        for i in y_lower:
            self.assertTrue(isinstance(i, numbers.Number))

        for i in y_upper:
            self.assertTrue(isinstance(i, numbers.Number))


### Run the tests
if __name__ == '__main__':
    unittest.main()
      