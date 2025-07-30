from pycaret.utils import version
import pandas as pd
data = pd.read_csv('insurance.csv')

from pycaret.regression import *
s = setup(data, target = 'charges')
xgboost = create_model('gbr')
save_model(xgboost, 'final-model')