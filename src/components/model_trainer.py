import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
    )
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr,test_arr):
        try:
            logging.info("Split training and test innput data")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "AdaBoost Regressor": AdaBoostRegressor(),
                "GradientBoosting Regressor": GradientBoostingRegressor(),
                "RandomForest Regressor" : RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "XGB Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "DecisionTree Regressor": DecisionTreeRegressor(),
                "KNeighbors Regressor": KNeighborsRegressor()
            }
            params={
                "AdaBoost Regressor":{
                    'learning_rate':[0.1,0.01,0.5,0.001],
                    # 'loss': ['linera','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoosting Regressor":{
                    # 'loss':['squared_error','huber','absolute_error','quantile'],
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'subsample': [0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error','friedman_mse'],
                    # 'max_feature':['auto','sqrt',log2],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "RandomForest Regressor":{
                    # 'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    # 'max_feature':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression": {},
                "XGB Regressor":{
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost Regressor":{
                    'depth': [6,8,10],
                    'learning_rate':[0.01,0.05,0.1],
                    # 'iteration':[10,20,30]
                },
                "DecisionTree Regressor": {
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson']
                    # 'splitter': ['best','random'],
                    # 'max_feature':['sqrt','log2'],
                },
                "KNeighbors Regressor":{
                    'n_neighbors':[5,7,9,11],
                    # 'weights':['uniform','distance'],
                    # 'algorithm': ['ball_tree','kd_tree','brute']
                }
            }

            model_report:dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models = models,param=params
            )

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info("Best found model for both taining and test data set")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e,sys)