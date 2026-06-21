import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


class Preprocessor:
    # Column definitions
    num_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary',
                'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate',
                'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
                'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt',
                'Total_EMI_per_month', 'Credit_History_Age' ,
                'Amount_invested_monthly', 'Monthly_Balance',]

    binary_text_cols = ['Payment_of_Min_Amount']

    binary_num_cols = ['Student Loan', 'Mortgage Loan', 'Debt Consolidation Loan',
                       'Payday Loan', 'Credit-Builder Loan', 'Personal Loan',
                       'Home Equity Loan', 'Auto Loan', 'Not Specified',]

    ordinal_cols = ['Credit_Mix']

    onehot_cols = ['Occupation', 'Payment_Behaviour']

    def __init__(self):
        self.column_transformer = None

    # Static: split dataset
    @staticmethod
    def split(df):
        x = df.drop(columns=['Credit_Score'])
        y = df['Credit_Score']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, stratify=y,)

        # Drop Month and Credit Utilization Ratio
        x_train = x_train.drop(columns=['Month'], errors='ignore')
        x_test  = x_test.drop(columns=['Month'], errors='ignore')

        x_train = x_train.drop(columns=['Credit_Utilization_Ratio'], errors='ignore')
        x_test  = x_test.drop(columns=['Credit_Utilization_Ratio'], errors='ignore')

        return x_train, x_test, y_train, y_test

    # Sklearn Pipeline 
    def fit(self, X, y=None):
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
        ])

        binary_text_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(categories=[['No', 'Yes']])),
        ])

        binary_num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
        ])

        ordinal_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(categories=[['Bad', 'Standard', 'Good']])),
        ])

        onehot_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ])

        self.column_transformer = ColumnTransformer([
            ('num', num_pipeline, self.num_cols),
            ('binary_text', binary_text_pipeline, self.binary_text_cols),
            ('binary_num', binary_num_pipeline, self.binary_num_cols),
            ('ordinal', ordinal_pipeline, self.ordinal_cols),
            ('onehot', onehot_pipeline, self.onehot_cols),
        ])

        self.column_transformer.fit(X)
        return self

    def transform(self, X):
        if self.column_transformer is None:
            raise RuntimeError("Please fit the preprocessor first.")
        return self.column_transformer.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


if __name__ == "__main__":
    df = pd.read_csv("ingested/data_D.csv")
    x_train, x_test, y_train, y_test = Preprocessor.split(df)

    prep = Preprocessor()
    x_train_transformed = prep.fit_transform(x_train)
    x_test_transformed  = prep.transform(x_test)

    print("Preprocessing Done!")
    print("x_train shape:", x_train_transformed.shape)
    print("x_test  shape:", x_test_transformed.shape)