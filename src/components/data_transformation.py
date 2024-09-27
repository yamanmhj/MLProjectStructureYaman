import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomeException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def return_pipelinetransformed_data(self):
        try:
            numerical_columns = ['a','b']
            categorical_columns = ['c','d','e','f']
            data_time_columns = []              
            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")), # fills missing value
                ("scaler",StandardScaler())
                ]
            )
            categorical_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            preprocessor=ColumnTransformer(  #
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",categorical_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomeException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path) 
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            
            preprocessed_data=self.return_pipelinetransformed_data()
            logging.info("Obtaining preprocessing object")

            target_column_name="target"
            numerical_columns = ["a", "b"]
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1) #drop target column
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            input_feature_pre_processed_train_array=preprocessed_data.fit_transform(input_feature_train_df)
            input_feature_pre_processed_test_array=preprocessed_data.transform(input_feature_test_df)


            train_array = np.c_[
                input_feature_pre_processed_train_array, np.array(target_feature_train_df)
            ]
            test_array = np.c_[input_feature_pre_processed_test_array, np.array(target_feature_test_df)]
            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=self.return_pipelinetransformed_data()
            )
            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomeException(e,sys)

