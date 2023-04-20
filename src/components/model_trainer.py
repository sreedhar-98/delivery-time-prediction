import os
import sys
from sklearn.linear_model import LinearRegression,RidgeCV,LassoCV,ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import numpy as np
from src.utils import train_evaluate_model,save_object

@dataclass
class ModelTrainerConfig:
    trainer_object_path=os.path.join('artifacts','best_model.pkl')

class modelTraining:
    def __init__(self):
        self.model_obj_path=ModelTrainerConfig()
    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("Model Training initiated")
            models_dict={
                'LinearRegression':LinearRegression(),
                'RidgeCV':RidgeCV(alphas=[0.1,0.5,1,2,5,8]),
                'LassoCV':LassoCV(alphas=[0.1,0.5,1,2,5,8]),
                'ElasticNetCV':ElasticNetCV(alphas=[0.1,0.5,1,2,5,8]),
                'DecisionTreeRegressor':DecisionTreeRegressor(),
                'RandomForestRegressor':RandomForestRegressor(),
                'AdaBoostRegressor':AdaBoostRegressor(),
                'GradientBoostingRegressor':GradientBoostingRegressor(),
                'XGBRegressor':XGBRegressor(gamma=0.05)
                }
            
            hyperparameters_dict={

                'DecisionTreeRegressor':{
                    #'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    #'min_samples_split':[2,4,6,8,10],
                    #'min_samples_leaf':[2,4,6,8,10],
                    'max_depth':[5,10,15,20,25,30],
                    #'max_features':['sqrt','log2']
                },
                'RandomForestRegressor':{
                    #'n_estimators':[100,200,300,400,500],
                    #'min_samples_split':[2,4,6,8,10],
                    #'min_samples_leaf':[2,4,6,8,10],
                    #'max_features':['sqrt','log2'],
                    'max_depth':[5,10,15,20,25,30]
                },
                "GradientBoostingRegressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "LinearRegression":{},
                "RidgeCV":{},
                "LassoCV":{},
                "ElasticNetCV":{},

                "XGBRegressor":{
                    'n_estimators':[100,200,300,400,500],
                    'max_depth':[5,10,15,20],
                    'learning_rate':[0.05,0.1,0.125,0.15,0.2],
                    'min_child_weight':[30,40,50]
                },
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }


            model_report,tuned_models_dict=train_evaluate_model(models_dict,hyperparameters_dict,train_arr,test_arr)
            logging.info("Training is completed and the report is generated.")
            logging.info(model_report)
            print('#'*135)
            print(model_report)
            scores=[]
            for model in model_report.index:
                scores.append((model,model_report.loc[model,('Test Dataset','r2 score')]))
        
            scores=sorted(scores,key=lambda x : x[1],reverse=True)

            print('#'*135)
            print('\n')
            print("Model with highest r2 score is {}({})".format(scores[0][0],scores[0][1]))
            logging.info(f"Model with highest r2 score is {scores[0][0]} ({scores[0][1]})")

            save_object(file_path=self.model_obj_path.trainer_object_path,obj=tuned_models_dict[scores[0][0]])


        except Exception as e:
            logging.info("Exception occured at model trainer")
            raise CustomException(e,sys)
