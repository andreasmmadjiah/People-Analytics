import os
import pandas as pd
import numpy as np

import time
import datetime
import calendar



from sklearn.metrics import mean_squared_error

import os

import sklearn
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.model_selection import KFold, StratifiedKFold


from sklearn import clone
from sklearn.linear_model import LinearRegression
'''
Ntr klo ga berat tiap model ama dkk jadiin class aja 
clas ...
    def build model
    
    def validation
    
    def tune
    
'''

def fast_build_model(num_cols,cat_cols,features,X,y,cv,model=LinearRegression()):

    num_transformer = Pipeline(steps=[
                                    ('imputer', SimpleImputer(strategy = 'constant',fill_value=0)),
                                    ('scaler', RobustScaler())
                                    ])

    cat_transformer = Pipeline(steps=[
                                    ('imputer', SimpleImputer(strategy='most_frequent')),
                                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                                    ])

    transformer = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ])


    main_pipeline = Pipeline(steps=[('transformer', transformer),
                          ('classifier', model)])
    
    skf = StratifiedKFold(n_splits=cv,random_state = 0)

    # oof validation
    oof_y_valid = []
    oof_y_valid_pred = []
    pipelines = []

    for cv,(train_index, val_index) in enumerate(skf.split(X,y)):
        start_fit = time.time()
        X_train = X.iloc[train_index,:]
        y_train = y.iloc[train_index]
        X_val = X.iloc[val_index,:]
        y_val = y.iloc[val_index]
        model = clone(main_pipeline)
        model.fit(X_train,y_train.values.ravel())
        pred = model.predict(X_val)
        oof_y_valid_pred.extend(pred)
        oof_y_valid.extend(y_val.values)

        pipelines.append(model)
        print(f'Fit iteration {cv} done in : {str(time.time()-start_fit)}')

    rmse = mean_squared_error(oof_y_valid,oof_y_valid_pred,squared=False)
    print('RMSE average :' , rmse)
    
    return pipelines



def tune_model(num_cols,cat_cols,features,pipelines,X,y,parameters, scorer=None,cv=4,n_iter=30,model=LinearRegression()): # random or grid-search
    
# To make it faster, use X train that already been pipelined
    cols_all = num_cols.copy()
    cols_all.extend(pipelines[0].named_steps['transformer'].transformers_[1][1].named_steps['onehot'].get_feature_names(cat_cols).tolist())


    permu_data_tune = pipelines[0].named_steps['transformer'].transform(X)
    permu_data_tune = pd.DataFrame(permu_data_tune,columns = cols_all)

    if scorer==None:
        scorer = make_scorer(rmse,squared=False)
    
    if method=='random':
        tune_search = RandomizedSearchCV(
            model,
            parameters,
            n_iter=30,
            scoring = scorer,
            cv = cv,
          random_state = 0,
          verbose = 1,
            n_jobs=-1

        )
    else:
        tune_search = GridSearchCV(
            model,
            parameters,
            scoring = scorer,
            cv = cv,
            verbose = 1,
            n_jobs=-1
        )
    import time # Just to compare fit times
    start = time.time()
    tune_search.fit(permu_data_tune, y.values.ravel())
    end = time.time()
    print("Tune Fit Time:", end - start)

    print('Best Score: %s' % tune_search.best_score_)
    print('Best Hyperparameters: %s' % tune_search.best_params_)
    return tune_search.best_params_
