
def data_transform_histgradientboost(dataset):
    import numpy as np
    import pandas as pd
    #dataset.drop_duplicates(subset = ['rid'],inplace=True)
    pd.options.mode.chained_assignment = None
    data2 = dataset[dataset['user_segment']!='No Segment']
    
    # X = data2[['user_visible_minority', 'user_smartphone_usage',
    #     'user_debit_tap_availability', 'user_age', 'user_income',
    #     'user_gender', 'user_region', 'user_education', 'user_lifestage',
    #     'user_immigrant_status', 'user_payment_preference',
    #     'user_banking_package', 'user_main_bank', 'user_main_debit_card',
    #     'user_main_credit_card']]

    # y = data2['user_segment']

    data2['user_segment'].replace({'The Achiever':'Heavy Credit',
            'The Maximizer':'Heavy Credit',
            'The Collector':'Heavy Credit',
            'The On-the-Goer':'Heavy Debit',
            'The Safekeeper':'Heavy Debit',
            'The Budgeter':'Heavy Cash',
            'The Traditionalist':'Heavy Cash'}, inplace=True)
    # binary variables, replacing with 1s and 0s
    data2['user_visible_minority'].replace(
                                    {
                                    "Not Visible Minority":0,
                                    "Visible Minority":1
                                    }, inplace=True)

    data2['user_smartphone_usage'].replace(
                                    {
                                    "Non-Smartphone Users":0,
                                    "Smartphone Users":1
                                    }, inplace=True)

    data2['user_debit_tap_availability'].replace(
                                    {
                                    "Tap Not Available":0,
                                    "DKNA":0,
                                    "Tap Available":1
                                    }, inplace=True)

    data2['user_age'].replace({
        '15 - 24':1,
        '25 - 34':2,
        '35 - 44':3,
        '45 - 54':4,
        '55 - 64':5,
        '65+': 6
    }, inplace=True)

    data2['user_income'].replace({
        '<$30k':1,
        '':1,
        '$30K - $59K':2,
        '$30k - $59k':2,
        '$60K - $99K':3,
        '$100k+':4 ,
        '$100K+':4
    }, inplace=True)


    data2['user_education'].replace({
        'High School or Less':1,
        'Some University/College':2,
        'University/College':3,
        'Advanced Degree':4
    }, inplace=True)

    data2['user_immigrant_status'].replace({
        'Less than 10 Years in Canada':1,
        '10 Years+ in Canada':2,
        'Canadian Born':3
    }, inplace=True)
    
    data2['trans_lp_type_collected'].replace(
                                {
                                "Other":1,
                                "Store Specific":1,
                                "No":0,
                                "General Points":1,
                                "Other Loyalty":1,
                                "Cash Back":1,
                                "SeparateCard":1,
                                "PaymentCard":1,
                                "Both":1,
                                "":0
                                },inplace = True)
    
    # # count number of online/ in-person and other transaction
    data3 = pd.get_dummies(data2,columns = ['trans_payment_type'])
    #data1['num_online'] = data1.groupby(['rid']).trans_payment_type_Online.transform(np.sum)
    data3.loc[:,'num_online'] = data3.groupby(['rid'])['trans_payment_type_Online'].transform('sum')
    # #data1['num_inperson'] = data1.groupby(['rid'])['trans_payment_type_In Person'].transform(np.sum)
    data3.loc[:,'num_inperson'] = data3.groupby(['rid'])['trans_payment_type_In Person'].transform('sum')
    # #data1['num_other'] = data1.groupby(['rid']).trans_payment_type_Other.transform(np.sum)
    data3.loc[:,'num_other'] = data3.groupby(['rid'])['trans_payment_type_Other'].transform('sum')
    # #data1['num_lp'] = data1.groupby(['rid']).trans_lp_type_collected.transform(np.sum)
    data3.loc[:,'num_lp'] = data3.groupby(['rid'])['trans_lp_type_collected'].transform('sum')
    
    data3['trans_payment_method'].replace(
                                    {
                                    "Mastercard Credit":"credit",
                                    "Visa Credit":"credit",
                                    "Reloadable/Pre-Paid":"credit",
                                    "Amex Credit":"credit",
                                    "Other Credit":"credit"
                                    },inplace= True)

    data3['trans_main_category'].replace(
                                    {
                                    "Grocery/Drug/Department/Warehouse Store":"grocery",
                                    "Restaurant/Bar/Coffee shop":"restaurant",
                                    "Convenience Store/Gas Station":"travel",
                                    "Transit/Travel/Parking":"travel",
                                    "Hardware/Home Improvement/Auto":"home",
                                    "Home Furnishing, Appliance, Decor, Home Office":"home",
                                    "Clothing/Jewellery/Shoes/Sporting Goods":"personal",
                                    "Electronics/Books/Music/Entertainment/Toys":"personal",
                                    "Miscellaneous Goods and Services":"Miscellaneous",
                                    "Health/Beauty/Drug Store":"Miscellaneous",
                                    "Other":"Miscellaneous",
                                    "Beer/Wine/Liquor":"Liquor"
                                    },inplace= True)

    # combine trans_payment_method_agg and trans_main_category_agg with /
    data3.loc[:,'trans_category_and_payment'] = data3[["trans_main_category","trans_payment_method"]].agg('/'.join,axis=1)

    # aggregate transaction amount by rid and trans_category_and_payment
    data3.loc[:,'total_amount'] = data3.groupby(['rid','trans_category_and_payment']).trans_amount.transform(np.sum)
    data3.sort_values(by=['rid','trans_category_and_payment'],inplace=True)
    
    # get distinct values of the dataframe based on rid and trans_category_and_payment
    data3.drop_duplicates(subset = ['rid','trans_category_and_payment'],inplace = True)
    
    # pivot table will generate new columns for trans_category_and_payment in user level
    data4 = data3.pivot_table('total_amount',['rid'],'trans_category_and_payment')
    # fill pivot table null values with 0
    data4.fillna(0,inplace= True)
    
    data5 = data3[['rid', 'trans_lp_type_collected', 
    'user_weight', 'user_weekly_factor', 
    'user_visible_minority', 'user_smartphone_usage',
    'user_debit_tap_availability',
    'user_segment','user_payment_preference', 'num_online', 'num_inperson', 'num_other', 'num_lp',
    'user_age','user_income','user_gender',
    'user_region','user_education','user_lifestage',
    'user_immigrant_status',
    'user_banking_package', 'user_main_bank',
    'user_main_debit_card','user_main_credit_card']]
    
    data5= data5.set_index('rid')
    data5['rid']=data5.index
    # get distinct values of the dataframe based on rid
    data5.drop_duplicates(subset = ['rid'],inplace = True)

    # combine user level data with pivot table
    data_final = pd.concat([data5,data4],axis = 1)



    lists = ['user_gender','user_region', 'user_lifestage',
            'user_payment_preference','user_banking_package', 
            'user_main_bank', 'user_main_debit_card',
            'user_main_credit_card']

    for i in lists:
        data_final[i] = data_final[i].str.lower()
    
    X = data_final.drop(columns=['user_segment'])
    y = data_final['user_segment']
    return X,y

def data_transform_XGB(dataset):
    import pandas as pd
    X,y = data_transform_histgradientboost(dataset)
    X_encoded = pd.get_dummies(X, columns = ['user_gender',
    'user_region', 'user_lifestage','user_payment_preference',
    'user_banking_package', 'user_main_bank', 'user_main_debit_card',
    'user_main_credit_card'])
    X_encoded['user_main_debit_card_td'] = 0

    import re
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    X_encoded.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_encoded.columns.values]
    y = y.replace({'Heavy Credit':0,
                'Heavy Debit':1,
                'Heavy Cash':2})
    
    return X_encoded,y