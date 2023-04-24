from flask import Flask,request,render_template,jsonify
from src.pipeline.predict_pipeline import CustomData,predictPipeline
import pandas as pd

application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('homepage.html')


@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('formpage.html')
    else:
        data=CustomData(
            Delivery_person_Age=int(request.form.get('Delivery_person_Age')),
            Delivery_person_Ratings=float(request.form.get('Delivery_person_Ratings')),
            Restaurant_latitude=float(request.form.get('Restaurant_latitude')),
            Restaurant_longitude=float(request.form.get('Restaurant_longitude')),
            Vehicle_condition=float(request.form.get('Vehicle_condition')),
            multiple_deliveries=float(request.form.get('multiple_deliveries')),
            Order_Date=request.form.get('Order_Date'),
            Time_Orderd=request.form.get('Time_Orderd'),
            Time_Order_picked=request.form.get('Time_Order_picked'),
            Weather_conditions=str(request.form.get('Weather_conditions')),
            Road_traffic_density=str(request.form.get('Road_traffic_density')),
            Festival=str(request.form.get('Festival')),
            City=str(request.form.get('City'))
        )
        data_df=data.get_data_as_df()
        data_df['Order_Date']=pd.to_datetime(data_df['Order_Date'],format='%Y-%m-%d')
        data_df['Order_Date']=data_df['Order_Date'].dt.strftime('%d-%m-%Y')
        pred_pipeline=predictPipeline()
        #print(data_df.columns)
        pred_val=pred_pipeline.predict(data_df)
        return render_template('resultpage.html',final_result=pred_val)
    

if __name__=="__main__":
    app.run(host='0.0.0.0',port=8080)

