# import necessary packages
import pandas as pd
from scr.data_transformation import data_transform_histgradientboost,data_transform_XGB
from scr.model_prediction import prediction
from scr.db_connection import mysql_conn


# MySql Local connection
connection = mysql_conn('localhost','ACP','root')

print("Reading Testing Dataset")
test = pd.read_sql("select * from ACP.interacdashboardfinalv2",connection)

# transform the input data
X,y = data_transform_histgradientboost(test)

print("HistGradientBoost Classifier prediction results")
prediction("histgradient.sav",X,y)


X_encoded,y = data_transform_XGB(test)
print("XGBoost Classifier prediction results")
prediction("xgb.sav",X_encoded,y)


print("RFECV XGBoost Classifier prediction results")
prediction("selector.sav",X_encoded,y)
