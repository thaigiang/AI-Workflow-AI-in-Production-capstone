"""
api tests
"""

try:
    import unittest2 as unittest
except:
    import unittest

import sys
import os
import requests
import numbers
import re
from ast import literal_eval
import numpy as np

port = 8050

try:
    requests.post('http://localhost:{}/'.format(port))
    server_available = True
except:
    server_available = False

    
## test class for the main window function
class ApiTest(unittest.TestCase):
    """
    test the essential functionality
    """

    @unittest.skipUnless(server_available,"local server is not running")
    def test_01_train(self):
        """
        test the train functionality
        """
      
        request_json = {'mode':'test'}
        r = requests.post('http://localhost:{}/train'.format(port),json=request_json)
        train_complete = re.sub("\W+","",r.text)
        self.assertEqual(train_complete,'true')
    
    @unittest.skipUnless(server_available,"local server is not running")
    def test_02_predict_empty(self):
        """
        ensure appropriate failure types
        """
    
        ## provide no data at all 
        r = requests.post('http://localhost:{}/predict'.format(port))
        self.assertEqual(re.sub('\n|"','',r.text),"[]")

        ## provide improperly formatted data
        r = requests.post('http://localhost:{}/predict'.format(port),json={"key":"value"})     
        self.assertEqual(re.sub('\n|"','',r.text),"[]")
    
    @unittest.skipUnless(server_available,"local server is not running")
    def test_03_predict(self):
        """
        test the predict functionality
        """
        country = 'all'
        year = 2018
        month = 2
        n_next = 20
        query_data = {'country': 'all',
                      'year': 2018,
                      'month': 2,
                      'n_next': 20
        }

        request_json = {'query':query_data,'mode':'test'}

        r = requests.post('http://localhost:{}/predict'.format(port),json=request_json)
        response = literal_eval(r.text)

        self.assertTrue(len(response['y_pred'])==n_next)
        self.assertTrue(len(response['y_lower'])==n_next)
        self.assertTrue(len(response['y_upper'])==n_next)

        for p in response['y_pred']:
            self.assertTrue(isinstance(p,numbers.Number))

        for p in response['y_lower']:
            self.assertTrue(isinstance(p,numbers.Number))

        for p in response['y_upper']:
            self.assertTrue(isinstance(p,numbers.Number))


    @unittest.skipUnless(server_available,"local server is not running")
    def test_04_logs(self):
        """
        test the log functionality
        """

        file_name = 'train-test.log'
        request_json = {'file':'train-test.log'}
        r = requests.get('http://localhost:{}/logs/{}'.format(port,file_name))

        with open(file_name, 'wb') as f:
            f.write(r.content)
        
        self.assertTrue(os.path.exists(file_name))

        if os.path.exists(file_name):
            os.remove(file_name)

        
### Run the tests
if __name__ == '__main__':
    unittest.main()
