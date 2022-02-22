def prediction(model_name,X,y):
    import pickle
    from dmba import classificationSummary
    loaded_model = pickle.load(open(model_name, 'rb'))
    if model_name == "xgb.sav":
        for feature_name in loaded_model.best_estimator_.get_booster().feature_names:
            if feature_name not in X.columns.tolist():
                X[model_name] = 0
    result = loaded_model.score(X, y)
    print("model accuracy on this dataset is: ", result)
    print(classificationSummary(y, loaded_model.predict(X)))