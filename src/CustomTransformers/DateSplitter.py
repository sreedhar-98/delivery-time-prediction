from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd

class DateSplitter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        # Access the column name(s) from the ColumnTransformer
        col_names = list(df.columns)
        for column_name in col_names:
            # Convert string column to datetime
            df[column_name] = pd.to_datetime(df[column_name],format='%d-%m-%Y')
            # Extract day, month, year
            col_names = [column_name + '_Day', column_name + '_Month', column_name + '_Year']
            df[col_names[0]] = df[column_name].dt.day
            df[col_names[1]] = df[column_name].dt.month
            df[col_names[2]] = df[column_name].dt.year
            # Drop original datetime column
            df.drop(column_name, axis=1, inplace=True)
        self.columns=df.columns
        return df
    
    def get_feature_names_out(self,names=None):
        return self.columns