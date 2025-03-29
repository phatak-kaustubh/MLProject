import os
import sys

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.exception import CustomException
from src.logger import logging

import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


#
from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer



@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("In initiate_data_ingestion()")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Reading the Dataset in to df')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info("***Train Test Split Initiate***")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=12)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("Ingestion of Data Done!!!")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )            
        except Exception as e:
            raise CustomException(e,sys)
            

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    #data_transformation.initiate_data_transformer(train_data,test_data)
    
    #for video 271
    train_arr,test_arr,_ = data_transformation.initiate_data_transformer(train_data,test_data)
    
    modeltrainer= ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
    




