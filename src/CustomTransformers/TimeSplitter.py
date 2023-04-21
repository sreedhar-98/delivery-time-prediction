from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd

class TimeSplitter(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        df=X.copy()
        col_names=list(df.columns)
        f=lambda x : int(24*x) if x<1 else x
        for column in col_names:
            col_names=[column+'_Hours',column+'_Minutes']
            time_dat=df[column].str.split(':')
            df[col_names[0]]=time_dat.str[0]
            df[col_names[1]]=time_dat.str[1]

            df[col_names[0]]=df[col_names[0]].astype(float)
            df[col_names[1]]=df[col_names[1]].astype(float)

            df[col_names[0]]=df[col_names[0]].apply(f)
            df.drop(column,axis=1,inplace=True)
        self.columns=df.columns
        return df
    
    def get_feature_names_out(self,names=None):
        return self.columns