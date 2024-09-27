import os
import sys
import sklearn
from src.logger import logging
from src.exception import CustomeException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    print("the training path is",train_data_path)
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')



class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
   
        print("Entered the data ingestion method or component")
        try:
            
            root_path=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__name__))))
            data_path =os.path.join(root_path,'Load_your_data_here','uber_data.csv')
            print(data_path)
            df = pd.read_csv(data_path)
            logging.info('Read the dataset as dtaframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            logging.info("artifacts created")

            
            
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)



            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomeException(e,sys)
           
        
if __name__ == "__main__":
    print("got in main")
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    final_training_data, final_testing_data = data_transformation.initiate_data_transformation(train_data,test_data)


    modeltrainer = ModelTrainer()
    modeltrainer.initiate_model_trainer(final_training_data,final_testing_data)