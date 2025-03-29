import sys
import os

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl") 

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformer_object(self):
        try:
            numerical_columns=[
                "reading_score",
                "writing_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"]
            
            #we create 2 pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),# to handle the missing valus
                    ("scaler",StandardScaler())# to handle the missing valus
                ]                
            )
            
            logging.info("Numerical Columns standard scaling Done!!!")
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("Categorical Columns encoding Done!!!")
            
            logging.info(f"Categorical Coulmns : {categorical_columns}")
            logging.info(f"Numerical Coulmns : {numerical_columns}")

            
            #combination of both the pipelines
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
                
            )
            logging.info("Return Preprocessor object!!")
            return preprocessor            
    
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformer(self,train_path,test_path):
        try:            
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("!!!!! Reading Train and TEst Data")
            
            logging.info("Obtaining Preprocessor Objecta ")
            
            
            preprocessor_obj = self.get_data_transformer_object()
            
            target_column_name="math_score"
            numerical_columns=["reading_score","writing_score"]
            
            # in the Train data set we split the dataset in to x_train and y_train
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            logging.info(f"Now Applying the Preprocessing ibject on the Teain  and test dataset")
            
            input_feature_train_array= preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_array= preprocessor_obj.transform(input_feature_test_df)
            
            logging.info(f"Done Fit_Transform and transform")
            #Why we are doing this??????????????????????????????????????
            '''
            np.c_[] merges them into a single NumPy array, making it easier to handle.
            Some ML models require NumPy arrays instead of Pandas DataFrames because NumPy is more memory-efficient and optimized for numerical computations.
            Libraries like Scikit-learn, TensorFlow, and PyTorch rely on NumPy for faster matrix operations and GPU acceleration.
            Pandas adds extra metadata (e.g., column names) that ML models don’t need, making NumPy the preferred choice for performance and compatibility.
            While some models accept DataFrames, converting to NumPy ensures better speed, parallelization, and deep learning compatibility
           
            '''
            train_arr = np.c_[
                input_feature_train_array,np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[
                input_feature_test_array,np.array(target_feature_test_df)
            ]
            logging.info("Saved preprocessing objects.")
            
            
            # used for saving the pkl file 
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
                )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
            