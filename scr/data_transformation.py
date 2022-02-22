def data_transform_histgradientboost(dataset):
    dataset.drop_duplicates(subset = ['rid'],inplace=True)
    data2 = dataset[dataset['user_segment']!='No Segment']
    
    X = data2[['user_visible_minority', 'user_smartphone_usage',
        'user_debit_tap_availability', 'user_age', 'user_income',
        'user_gender', 'user_region', 'user_education', 'user_lifestage',
        'user_immigrant_status', 'user_payment_preference',
        'user_banking_package', 'user_main_bank', 'user_main_debit_card',
        'user_main_credit_card']]

    y = data2['user_segment']

    y.replace({'The Achiever':'Heavy Credit',
            'The Maximizer':'Heavy Credit',
            'The Collector':'Heavy Credit',
            'The On-the-Goer':'Heavy Debit',
            'The Safekeeper':'Heavy Debit',
            'The Budgeter':'Heavy Cash',
            'The Traditionalist':'Heavy Cash'}, inplace=True)
    # binary variables, replacing with 1s and 0s
    X['user_visible_minority'].replace(
                                    {
                                    "Not Visible Minority":0,
                                    "Visible Minority":1
                                    }, inplace=True)

    X['user_smartphone_usage'].replace(
                                    {
                                    "Non-Smartphone Users":0,
                                    "Smartphone Users":1
                                    }, inplace=True)

    X['user_debit_tap_availability'].replace(
                                    {
                                    "Tap Not Available":0,
                                    "DKNA":0,
                                    "Tap Available":1
                                    }, inplace=True)

    X['user_age'].replace({
        '15 - 24':1,
        '25 - 34':2,
        '35 - 44':3,
        '45 - 54':4,
        '55 - 64':5,
        '65+': 6
    }, inplace=True)

    X['user_income'].replace({
        '<$30k':1,
        '':1,
        '$30K - $59K':2,
        '$30k - $59k':2,
        '$60K - $99K':3,
        '$100k+':4 ,
        '$100K+':4
    }, inplace=True)


    X['user_education'].replace({
        'High School or Less':1,
        'Some University/College':2,
        'University/College':3,
        'Advanced Degree':4
    }, inplace=True)

    X['user_immigrant_status'].replace({
        'Less than 10 Years in Canada':1,
        '10 Years+ in Canada':2,
        'Canadian Born':3
    }, inplace=True)


    lists = ['user_gender',
        'user_region', 'user_lifestage','user_payment_preference',
        'user_banking_package', 'user_main_bank', 'user_main_debit_card',
        'user_main_credit_card']

    for i in lists:
        X[i] = X[i].str.lower()
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