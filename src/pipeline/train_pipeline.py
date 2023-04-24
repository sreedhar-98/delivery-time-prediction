from src.components.data_ingestion import DataConfig
from src.components.data_transformation import DataTransformer
from src.components.model_trainer import modelTraining

dc=DataConfig()
train_path,test_path=dc.initiateIngestion()
dt=DataTransformer()
train_arr,test_arr,_=dt.initiate_data_transformation(train_data_path=train_path,test_data_path=test_path)
mt=modelTraining()
mt.initiate_model_training(train_arr,test_arr)
# print(train_arr)