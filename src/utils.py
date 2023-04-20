import os
import pickle
from src.exception import CustomException
import sys
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
import sys

def save_object(file_path,obj): 
     try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
     except Exception as e:
        raise CustomException(e, sys)
     
def train_evaluate_model(models_dict,hyperparameters_dict,train_arr,test_arr):
    X_train,X_test,y_train,y_test=train_arr[:,:-1],test_arr[:,:-1],train_arr[:,-1],test_arr[:,-1]
    metrics=['r2 score','Root Mean Squared Error','Mean Absolute Error']
    cols=pd.MultiIndex.from_product([['Training Dataset','Test Dataset'],metrics])
    report=pd.DataFrame(index=models_dict.keys(),columns=cols)
    try:
      logging.info("Model Training Started")
      for model in models_dict:
         trainer=models_dict[model]
         params=hyperparameters_dict[model]

         gs=GridSearchCV(estimator=trainer,param_grid=params,cv=5,scoring='r2',n_jobs=-1)
         gs.fit(X_train,y_train)

         tuned_trainer=gs.best_estimator_

         models_dict[model]=tuned_trainer

        
         y_pred_train=tuned_trainer.predict(X_train)
         y_pred_test=tuned_trainer.predict(X_test)

         mse_train=mean_squared_error(y_train,y_pred_train)
         mse_test=mean_squared_error(y_test,y_pred_test)

         mae_train=mean_absolute_error(y_train,y_pred_train)
         mae_test=mean_absolute_error(y_test,y_pred_test)

         report.loc[model,('Training Dataset','Mean Absolute Error')]=mae_train
         report.loc[model,('Test Dataset','Mean Absolute Error')]=mae_test

         rmse_train=np.sqrt(mse_train)
         rmse_test=np.sqrt(mse_test)

         report.loc[model,('Training Dataset','Root Mean Squared Error')]=rmse_train
         report.loc[model,('Test Dataset','Root Mean Squared Error')]=rmse_test

         r2_score_train=r2_score(y_train,y_pred_train)
         r2_score_test=r2_score(y_test,y_pred_test)

         report.loc[model,('Training Dataset','r2 score')]=r2_score_train
         report.loc[model,('Test Dataset','r2 score')]=r2_score_test
    
      return report,models_dict
    except Exception as e:
       logging.info("Error occured while training the model")
       raise CustomException(e,sys)      

def load_object(file_path):
   try:
      with open(file_path,'rb') as f:
         return pickle.load(f)
   except Exception as e:
      logging.info("Exception occured while loading the pickle file")
      raise CustomException(e,sys)
        

        