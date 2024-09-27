import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.model_selection import GridSearchCV
from src.exception import CustomeException
from sklearn.metrics import r2_score




def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        print("the file path in save_obj is",dir_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomeException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for model_name, model in models.items():
            search_space = param[model_name]

            grid_search = GridSearchCV(model,search_space,cv=3)
            grid_search.fit(X_train,y_train)

            model.set_params(**grid_search.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
        return report

    except Exception as e:
        raise CustomeException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomeException(e, sys)