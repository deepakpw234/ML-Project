import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationconfig:
    preprocerssor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")
    # print(f"20: {preprocerssor_obj_file_path}")

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationconfig()
        # print(f"25: {self.data_transformation_config}")

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ["reading_score", "writing_score"]
            category_columns = [
                'gender', 'race_ethnicity', 
                'parental_level_of_education', 'lunch', 
                'test_preparation_course'
                ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            # print(f"45: {num_pipeline}")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            # print(f"54: {cat_pipeline}")

            logging.info(f"Numerical columns : {numerical_columns}")

            logging.info(f"Categorical columns :{category_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,category_columns)
                ]
            )
            # print(f"66: {preprocessor}")
            logging.info("Data transformation is completed")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            # print(f"78: {train_path}")
            # print(f"79: {test_path}")


            logging.info("Read train and test data completed")

            logging.info("obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()
            # print(f"87: {preprocessing_obj}")

            target_column_name = 'math_score'
            numerical_columns = ["reading_score", "writing_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            # print(f"102: {input_feature_train_arr}")
            # print(f"103: {input_feature_test_arr}")
            
            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            # print(f"109: {train_arr}")
            # print(f"110: {np.array(target_feature_train_df)}")
            

            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            # print(f"117: {test_arr}")
            # print(f"118: {np.array(target_feature_test_df)}")

            logging.info("Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocerssor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocerssor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)