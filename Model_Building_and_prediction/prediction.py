# import necessary packages
import pandas as pd
from scr.data_transformation_trans import data_transform_histgradientboost,data_transform_XGB
from scr.model_prediction import prediction
from scr.db_connection import mysql_conn

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

# MySql Local connection
connection = mysql_conn('localhost','ACP','root')

print("Reading Testing Dataset")
test = pd.read_sql("select * from ACP.interacdashboardfinalv2",connection)

############################################################################################################################################
# Model 1A

# transform the input data
X,y = data_transform_histgradientboost(test)
X1= X[demo_bank_col]
X1 = X1.drop(columns = ['user_payment_preference'])

print("HistGradientBoost Classifier Model 1A prediction results")
prediction("histgradient_1A.sav",X1,y)

X,y = data_transform_XGB(test)
columns = ['user_visible_minority', 'user_smartphone_usage',
       'user_debit_tap_availability', 'user_age', 'user_income','user_education',
       'user_immigrant_status','user_gender_female',
       'user_gender_male', 'user_gender_other', 'user_region_alberta',
       'user_region_british columbia', 'user_region_east',
       'user_region_man/sask', 'user_region_north', 'user_region_ontario',
       'user_region_quebec', 'user_lifestage_couple no kids',
       'user_lifestage_couple with older kids',
       'user_lifestage_couple with younger kids',
       'user_lifestage_empty nesters', 'user_lifestage_single no kids',
       'user_lifestage_single with kids', 'user_payment_preference_cash',
       'user_payment_preference_credit', 'user_payment_preference_debit',
       'user_payment_preference_mixed','user_banking_package_dkna',
       'user_banking_package_limited', 'user_banking_package_unlimited',
       'user_main_bank_bmo', 'user_main_bank_cibc',
       'user_main_bank_desjardins', 'user_main_bank_other',
       'user_main_bank_pc/simplii', 'user_main_bank_rbc',
       'user_main_bank_scotiabank', 'user_main_bank_tangerine',
       'user_main_bank_td', 'user_main_debit_card_bmo',
       'user_main_debit_card_cibc', 'user_main_debit_card_desjardins',
       'user_main_debit_card_other', 'user_main_debit_card_pc/simplii',
       'user_main_debit_card_rbc', 'user_main_debit_card_scotiabank',
       'user_main_debit_card_tangerine', 'user_main_debit_card_td',
       'user_main_credit_card_amex credit',
       'user_main_credit_card_mastercard credit',
       'user_main_credit_card_no credit card',
       'user_main_credit_card_other credit',
       'user_main_credit_card_visa credit']

X1= X[columns]
X1 = X1.drop(['user_payment_preference_cash',
       'user_payment_preference_credit', 'user_payment_preference_debit',
       'user_payment_preference_mixed'],axis = 1)

print("XGB Classifier Model 1A prediction results")
prediction("xgb_1A.sav",X1,y)

print("XGB selector Classifier Model 1A prediction results")
prediction("selector_1A.sav",X1,y)


############################################################################################################################################
# Model 1B

X,y = data_transform_histgradientboost(test)
X1= X[demo_bank_col]

print("HistGradientBoost Classifier Model 1B prediction results")
prediction("histgradient_1B.sav",X1,y)

X,y = data_transform_XGB(test)
columns = ['user_visible_minority', 'user_smartphone_usage',
       'user_debit_tap_availability', 'user_age', 'user_income','user_education',
       'user_immigrant_status','user_gender_female',
       'user_gender_male', 'user_gender_other', 'user_region_alberta',
       'user_region_british columbia', 'user_region_east',
       'user_region_man/sask', 'user_region_north', 'user_region_ontario',
       'user_region_quebec', 'user_lifestage_couple no kids',
       'user_lifestage_couple with older kids',
       'user_lifestage_couple with younger kids',
       'user_lifestage_empty nesters', 'user_lifestage_single no kids',
       'user_lifestage_single with kids', 'user_payment_preference_cash',
       'user_payment_preference_credit', 'user_payment_preference_debit',
       'user_payment_preference_mixed','user_banking_package_dkna',
       'user_banking_package_limited', 'user_banking_package_unlimited',
       'user_main_bank_bmo', 'user_main_bank_cibc',
       'user_main_bank_desjardins', 'user_main_bank_other',
       'user_main_bank_pc/simplii', 'user_main_bank_rbc',
       'user_main_bank_scotiabank', 'user_main_bank_tangerine',
       'user_main_bank_td', 'user_main_debit_card_bmo',
       'user_main_debit_card_cibc', 'user_main_debit_card_desjardins',
       'user_main_debit_card_other', 'user_main_debit_card_pc/simplii',
       'user_main_debit_card_rbc', 'user_main_debit_card_scotiabank',
       'user_main_debit_card_tangerine', 'user_main_debit_card_td',
       'user_main_credit_card_amex credit',
       'user_main_credit_card_mastercard credit',
       'user_main_credit_card_no credit card',
       'user_main_credit_card_other credit',
       'user_main_credit_card_visa credit']

X1= X[columns]

print("XGB Classifier Model 1B prediction results")
prediction("xgb_1B.sav",X1,y)

print("XGB selector Classifier Model 1B prediction results")
prediction("selector_1B.sav",X1,y)

############################################################################################################################################
# Model 2A

# transform the input data
X,y = data_transform_histgradientboost(test)
X1= X[interac_tran_col]

print("HistGradientBoost Classifier Model 2A prediction results")
prediction("histgradient_2A.sav",X1,y)

X,y = data_transform_XGB(test)

X1= X[interac_tran_col]

print("XGB Classifier Model 2A prediction results")
prediction("xgb_2A.sav",X1,y)

print("XGB selector Classifier Model 2A prediction results")
prediction("selector_2A.sav",X1,y)

print("KNN Classifier Model 2A prediction results")
prediction("KNN_2A.sav",X1,y)


############################################################################################################################################
# Model 2B

# transform the input data
X,y = data_transform_histgradientboost(test)
X1= X[all_tran_col]

print("HistGradientBoost Classifier Model 2B prediction results")
prediction("histgradient_2B.sav",X1,y)

X,y = data_transform_XGB(test)

X1= X[all_tran_col]

print("XGB Classifier Model 2B prediction results")
prediction("xgb_2B.sav",X1,y)

print("XGB selector Classifier Model 2B prediction results")
prediction("selector_2B.sav",X1,y)

print("KNN Classifier Model 2B prediction results")
prediction("KNN_2B.sav",X1,y)

