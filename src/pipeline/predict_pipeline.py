import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from src.utils import load_object
import json

class predictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            with open('src/feature_engineering_artifacts/features_dict.json','r') as f:
                cols=json.load(f)
            
            preprocessor_object_path=os.path.join('artifacts','transformer.pkl')
            model_object_path=os.path.join('artifacts','best_model.pkl')

            preprocessor=load_object(preprocessor_object_path)
            logging.info("PreProcessor loaded successfully")
            model=load_object(model_object_path)
            logging.info("Model loaded successfully")

            data_scaled=pd.DataFrame(preprocessor.transform(features),columns=preprocessor.get_feature_names_out())
            data_scaled.drop(cols['dropped_cols'],axis=1,inplace=True)
            pred_val=model.predict(data_scaled)

            logging.info("Prediction Successful!!")
            return int(pred_val)
        except Exception as e:
            logging.info("Exception occured at predictPipeline class")
            raise CustomException(e,sys)
    
class CustomData:
    def __init__(self,
                 Delivery_person_Age:int,
                 Delivery_person_Ratings:float,
                 Restaurant_latitude:float,
                 Restaurant_longitude:float,
                 Vehicle_condition:int,
                 multiple_deliveries:float,
                 Order_Date:str,
                 Time_Orderd:str,
                 Time_Order_picked:str,
                 Weather_conditions:str,
                 Road_traffic_density:str,
                 Festival:str,
                 City:str
                 ):
        self.Delivery_person_Age=Delivery_person_Age
        self.Delivery_person_Ratings=Delivery_person_Ratings
        self.Restaurant_latitude=Restaurant_latitude
        self.Restaurant_longitude=Restaurant_longitude
        self.Vehicle_condition=Vehicle_condition
        self.multiple_deliveries=multiple_deliveries
        self.Order_Date=Order_Date
        self.Time_Orderd=Time_Orderd
        self.Time_Order_picked=Time_Order_picked
        self.Weather_conditions=Weather_conditions
        self.Road_traffic_density=Road_traffic_density
        self.Festival=Festival
        self.City=City

    def get_data_as_df(self):
        try:
            custom_data_input_dict = {
                'Delivery_person_Age':[self.Delivery_person_Age],
                'Delivery_person_Ratings':[self.Delivery_person_Ratings],
                'Restaurant_latitude':[self.Restaurant_latitude],
                'Restaurant_longitude':[self.Restaurant_longitude],
                'Vehicle_condition':[self.Vehicle_condition],
                'multiple_deliveries':[self.multiple_deliveries],
                'Order_Date':[self.Order_Date],
                'Time_Orderd':[self.Time_Orderd],
                'Time_Order_picked':[self.Time_Order_picked],
                'Weather_conditions':[self.Weather_conditions],
                'Road_traffic_density':[self.Road_traffic_density],
                'Festival':[self.Festival],
                'City':[self.City]
            }
            df=pd.DataFrame(custom_data_input_dict)
            logging.info("Data Frame created")
            return df
        except Exception as e:
            logging.info("Exception occured in prediction pipeline Custom Data class")
            raise CustomException(e,sys)

#Code to test the predict pipeline
# pp=predictPipeline()
# cd=CustomData(32,4.2,25.3210,81.2564,0,2.0,'14-02-2021','17:15','17:30',"Sandstorms","Medium","No","Metropolitian")
# df=cd.get_data_as_df()
# print(df)
# # res=pp.predict(df)
# # print(res)