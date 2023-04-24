import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import pandas as pd
import sys
import numpy as np
from src.utils import save_object
import json
import re
from src.CustomTransformers.DateSplitter import DateSplitter
from src.CustomTransformers.TimeSplitter import TimeSplitter

@dataclass
class TransformationConfig:
    transformer_path=os.path.join('artifacts','transformer.pkl')
class DataTransformer:
    def __init__(self):
        self.transformer_path=TransformationConfig()
    def get_transform_object(self):
        logging.info("Data Transformation object fetch initiated")
        try:
            
            with open('src/feature_engineering_artifacts/features_dict.json','r') as f:
                cols=json.load(f)
            with open('src/feature_engineering_artifacts/encoding.json','r') as f:
                encoding=json.load(f)
            
            numerical_cols=cols['numerical_cols']
            categorical_cols=cols['categorical_cols']
            date_cols=cols['date_cols']
            time_cols=cols['time_cols']

            Weather_conditions=encoding['Weather_conditions']
            Road_traffic_density=encoding['Road_traffic_density']
            Festival=encoding['Festival']
            City=encoding['City']
            
            num_pipeline=Pipeline( 
                steps=[  
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
                ]
            )

            date_pipeline=Pipeline(
                steps=[
                ('date_splitter',DateSplitter(format='%d-%m-%Y')),
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('scaler',StandardScaler())
                ]
            )

            time_pipeline=Pipeline(
                steps=[
                ('time_splitter',TimeSplitter()),
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('scaler',StandardScaler())
                ]
            )


            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinal_encoder',OrdinalEncoder(categories=[Weather_conditions,Road_traffic_density,Festival,City])),
                ('scaler',StandardScaler())
                ]
            )
            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('date_pipeline',date_pipeline,date_cols),
                ('time_pipeline',time_pipeline,time_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
                ],verbose_feature_names_out=False)  
            return preprocessor
        except Exception as e:
            logging.info("Exception occured at Data Transformation step")
            raise CustomException(e,sys)



    def initiate_data_transformation(self,train_data_path,test_data_path):
        logging.info("Data Transformation initiated")
        try:
            train_data=pd.read_csv(train_data_path)
            test_data=pd.read_csv(test_data_path)
            
            with open('src/feature_engineering_artifacts/features_dict.json','r') as f:
                cols=json.load(f)
            feature_col=cols['target_col']
            dropped_cols=cols['dropped_cols']

            X_train=train_data.drop(feature_col,axis=1)
            y_train=train_data[feature_col]

            X_test=test_data.drop(feature_col,axis=1)
            y_test=test_data[feature_col]

            preprocessor=self.get_transform_object()
            logging.info("Applying preprocessing")
            X_train=pd.DataFrame(preprocessor.fit_transform(X_train),columns=preprocessor.get_feature_names_out())
            X_test=pd.DataFrame(preprocessor.transform(X_test),columns=preprocessor.get_feature_names_out())

            X_train.drop(dropped_cols,axis=1,inplace=True)
            X_test.drop(dropped_cols,axis=1,inplace=True)
            
            #print(preprocessor.get_feature_names_out())
            
            train_arr=np.c_[X_train.to_numpy(),np.array(y_train)]
            test_arr=np.c_[X_test.to_numpy(),np.array(y_test)]

            save_object(file_path=self.transformer_path.transformer_path,obj=preprocessor)
            logging.info("Pickle file saved successfully")

            return train_arr,test_arr,self.transformer_path
        except Exception as e:
            logging.info("Exception occured at data transformation")
            raise CustomException(e,sys)