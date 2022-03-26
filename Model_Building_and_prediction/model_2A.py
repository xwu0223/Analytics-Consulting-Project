# import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.metrics import balanced_accuracy_score
from dmba import classificationSummary
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import warnings
import pickle
from scr.data_transformation_trans import data_transform_histgradientboost,data_transform_XGB
from scr.db_connection import mysql_conn


# MySql Local connection
connection = mysql_conn('localhost','ACP','root')
    
print("Reading Training Dataset")
data = pd.read_sql("select * from ACP.interac_dashboard",connection)


# Only year 2019 and 2020 will be considered due to the fact of COVID- 19 and target variable labelling
data1 = data[data['trans_year'].isin([2019,2020])==True]

demo_bank_col = ['user_visible_minority', 'user_smartphone_usage',
       'user_debit_tap_availability', 'user_age', 'user_income',
       'user_gender', 'user_region', 'user_education', 'user_lifestage',
       'user_immigrant_status', 'user_payment_preference',
       'user_banking_package', 'user_main_bank', 'user_main_debit_card',
       'user_main_credit_card']

interac_tran_col = ['Liquor/Interac', 'Miscellaneous/Interac', 'grocery/Interac',
       'home/Interac','personal/Interac','restaurant/Interac', 'travel/Interac']

all_tran_col = ['Liquor/Cash', 'Liquor/Interac', 'Liquor/Other', 'Liquor/credit',
       'Miscellaneous/Cash', 'Miscellaneous/Interac', 'Miscellaneous/Other',
       'Miscellaneous/credit', 'grocery/Cash', 'grocery/Interac',
       'grocery/Other', 'grocery/credit', 'home/Cash', 'home/Interac',
       'home/Other', 'home/credit', 'personal/Cash', 'personal/Interac',
       'personal/Other', 'personal/credit', 'restaurant/Cash',
       'restaurant/Interac', 'restaurant/Other', 'restaurant/credit',
       'travel/Cash', 'travel/Interac', 'travel/Other', 'travel/credit']

############################################################################################################################################
############################################## Model 2A(Internal Transaction Mirror)########################################################
# HistGradientBoost
############################################################################################################################################
# X,y = data_transform_histgradientboost(data1)
# X1= X[interac_tran_col]
# categorical_columns_selector = selector(dtype_include=object)
# categorical_columns = categorical_columns_selector(X1)
# categorical_columns
# categorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value",
#                                           unknown_value=-1)

# preprocessor = ColumnTransformer([
#     ('cat_preprocessor', categorical_preprocessor, categorical_columns)],
#     remainder='passthrough', sparse_threshold=0)

# X_train, X_test,y_train,y_test = train_test_split(X1,y,test_size=0.2,
#                                                     random_state=1234,stratify=y)

# hist_model = Pipeline([
#     ("preprocessor", preprocessor),
#     ("classifier",
#      HistGradientBoostingClassifier(random_state=42))])

# # Hist Gradient Boosting Classifier Training
# print("training Hist Gradient Boosting Classifier")
# param_grid = {
#     'classifier__learning_rate': (0.01, 0.1, 1, 10),
#     'classifier__max_leaf_nodes': (3, 10, 30),
#     'classifier__max_iter': [1000,1200,1500],
#     'classifier__learning_rate': [0.1],
#     'classifier__max_depth' : [5,10,20,25, 50, 75],
#     'classifier__l2_regularization': [1.5],
#     'classifier__scoring': ['f1_micro']}
# model_grid_search = GridSearchCV(hist_model, param_grid=param_grid,
#                                  n_jobs=2, cv=5)
# model_grid_search.fit(X_train, y_train)
# print(f"The best set of parameters is: "
#       f"{model_grid_search.best_params_}")
# accuracy = model_grid_search.score(X_train, y_train)
# print(
#     f"The train accuracy score of the grid-searched pipeline is: "
#     f"{accuracy:.2f}"
# )

# accuracy = model_grid_search.score(X_test, y_test)
# print(
#      f"The test accuracy score of the grid-searched pipeline is: "
#      f"{accuracy:.2f}"
#  )
# classificationSummary(y_train, model_grid_search.predict(X_train))
# classificationSummary(y_test, model_grid_search.predict(X_test))

# filename = 'histgradient_2A.sav'
# pickle.dump(model_grid_search,open(filename,'wb'))

# ############################################################################################################################################

# # XGBoost
# ############################################################################################################################################
# X_encoded,y = data_transform_XGB(data1)


# X1= X_encoded[interac_tran_col]

# X_train,X_test,y_train,y_test = train_test_split(X1,y,test_size=0.2,random_state = 1234,stratify=y)

# warnings.filterwarnings("ignore")
# clf_xgb = xgb.XGBClassifier(objective = 'multi:softprob', use_label_encoder=False, missing=0, seed=42)

# param_grid = {
#     'max_depth' :[3,4,5],
#     'learning_rate':[0.1,0.01,0.05],
#     'gamma':[0,0.25,1.0],
#     'reg_lambda':[0,1.0,10.0]
# }
# print("training first set of hyperparameter")
# model_grid_search = GridSearchCV(clf_xgb, param_grid=param_grid,
#                                  n_jobs=-1, cv=5)
# model_grid_search.fit(X_train, y_train,eval_metric='mlogloss')
# print(f"The best set of parameters is: "
#       f"{model_grid_search.best_params_}")
# balanced_accuracy_score(y_test, model_grid_search.predict(X_test))
# from dmba import classificationSummary
# classificationSummary(y_train, model_grid_search.predict(X_train))
# classificationSummary(y_test, model_grid_search.predict(X_test))
# clf_xgb = xgb.XGBClassifier(objective = 'multi:softprob', use_label_encoder=False, missing=0, seed=42)

# param_grid = {
#     'max_depth' :[3,4,5],
#     'learning_rate':[0.5,0.1,0.05],
#     'gamma':[0,0.25,0.5,1],
#     'reg_lambda':[10,20,30,40,50,70,90,100]
# }

# print("training second set of hyperparameter")

# model_grid_search = GridSearchCV(clf_xgb, param_grid=param_grid,
#                                  n_jobs=-1, cv=5)
# model_grid_search.fit(X_train, y_train,eval_metric='mlogloss')
# print(f"The best set of parameters is: "
#       f"{model_grid_search.best_params_}")
# balanced_accuracy_score(y_test, model_grid_search.predict(X_test))
# from dmba import classificationSummary
# classificationSummary(y_train, model_grid_search.predict(X_train))
# classificationSummary(y_test, model_grid_search.predict(X_test))
# plot_confusion_matrix(model_grid_search,
#                       X_test,
#                       y_test,
#                       values_format='d')

# filename = 'xgb_2A.sav'
# pickle.dump(model_grid_search,open(filename,'wb'))

# ############################################################################################################################################

# # RFECV
# X_encoded,y = data_transform_XGB(data1)

# X1= X_encoded[interac_tran_col]
# X_train,X_test,y_train,y_test = train_test_split(X1,y,test_size=0.2,random_state = 42,stratify=y)
# #choose estimator/model type for Recursive feature elimination and cross valiation
# estimator = xgb.XGBClassifier(gamma =1, learning_rate= 0.5, max_depth= 4, reg_lambda= 50, eval_metric='mlogloss',use_label_encoder =False)

# print("training RFECV")
# selector = RFECV(estimator, step=1, min_features_to_select=1,scoring="accuracy", cv=10)

# #fit the model, get a rank of the variables, and a matrix of the selected X variables
# selector = selector.fit(X_train, y_train.values.flatten())

# #PLot # of features selected vs. Model Score
# plt.figure()
# plt.title('XGB CV score vs No of Features')
# plt.xlabel("Number of features selected")
# plt.ylabel("accuracy")
# plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
# plt.show()

# # #get rank of X model features
# # rank = selector.ranking_
# # #Subset features to those selected by recursive feature elimination
# # #X_train_scaled = X_train_scaled[:,selector.support_ ] 

# # y_pred = selector.predict(X_train_scaled)
# plot_confusion_matrix(selector,
#                       X_test,
#                       y_test,
#                       values_format='d')
# pd.set_option('display.max_rows', 100)
# print(pd.DataFrame(
#     zip(X_train.columns, abs(selector.estimator_.feature_importances_)),
#     columns=["feature", "weight"],
# ).sort_values("weight",ascending=False).reset_index(drop=True))

# print(classificationSummary(y_train, selector.predict(X_train)))
# print(classificationSummary(y_test, selector.predict(X_test)))

# filename = 'selector_2A.sav'
# pickle.dump(selector,open(filename,'wb'))

############################################################################################################################################
# KNN
X_encoded,y = data_transform_XGB(data1)

X1= X_encoded[interac_tran_col]
X_train,X_test,y_train,y_test = train_test_split(X1,y,test_size=0.2,random_state = 42,stratify=y)
knn = KNeighborsClassifier()
k_range = list(range(1, 31))
param_grid = dict(n_neighbors=k_range)
# defining parameter range
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False,verbose=1)

# fitting the model for grid search
grid_search=grid.fit(X_train, y_train)
print(classificationSummary(y_train, grid.predict(X_train)))
print(classificationSummary(y_test, grid.predict(X_test)))

filename = 'KNN_2A.sav'
pickle.dump(grid,open(filename,'wb'))