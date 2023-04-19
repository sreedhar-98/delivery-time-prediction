import os
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import sys

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path=os.path.join('artifacts','raw.csv')

class DataConfig:
    def __init__(self):
        self.ingestionConfig=DataIngestionConfig()
    def initiateIngestion(self):
        logging.info("Data Ingestion initiated")
        try:
            raw_data=pd.read_csv(os.path.join('notebooks\datasets\delivery-time-prediction-cleaned.csv'))
            logging.info("Raw data read as pandas DataFrame")
            os.makedirs(os.path.dirname(self.ingestionConfig.raw_data_path),exist_ok=True)
            raw_data.to_csv(self.ingestionConfig.raw_data_path,index=False)
            logging.info("Train test split")
            train_data,test_data=train_test_split(raw_data,test_size=0.3,random_state=42)
            train_data.to_csv(self.ingestionConfig.train_data_path,index=False)
            test_data.to_csv(self.ingestionConfig.test_data_path,index=False)
            logging.info("Data ingestion successful")
            return self.ingestionConfig.train_data_path,self.ingestionConfig.test_data_path
        except Exception as e:
            logging.info("Exception at data ingestion")
            raise CustomException(e,sys)