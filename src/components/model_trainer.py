import os
import sys
from dataclasses import dataclass

#from Catboost import -> install error so commented

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split train and test data")
            X_train,y_train,X_test,y_test=(
                # '''
                # X_train # # ...| y_train
                # ...
                # ...
                # ..
                # .
                # X_test # # ... | y_test
                # '''
                train_array[:,:-1],#All Coumns , take out last column and save everythinh in X_train
                train_array[:,-1],#All rows,last cloumn to store in y_train
                test_array[:,:-1],
                test_array[:,-1]
            )
            # now we create a Dictionary of Models to try out on the dataset a
            #and see which worls well
            models = {
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "LinearRegression":LinearRegression(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                #Not done hyper parameter tuning 
                # to try it by yourself
            }
            
            #-evaluate_model function in utils 
            #get the r2 score for all the models in the form of dictionary
            model_report:dict=evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models)
            
            #Will get the best model score from the dictionry
            best_model_score = max(sorted(model_report.values()))
            
            #get the nme of best model from the key of the disctionary
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model=models[best_model_name]
            
            if(best_model_score)<0.6:
                raise CustomException("#####No Model performing better that 60% ")
            logging.info("Best Model Founf on Both Train and Test datast")
            
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            r2_sco= r2_score(y_test,predicted)
            
            return r2_sco
            
            
        except Exception as e:
            raise CustomException(e,sys)
            
        
