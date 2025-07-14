# Preprocess for the data
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# All the basic_cleaning can be found in eda.ipynb for detailed explanation.
def basic_cleaning(df):
    df['Age'] = df['Age'].str.extract(r'(\d+)').astype(int)
    df = df[(df['Age'] >= 18) & (df['Age'] <= 100)]

    df['Occupation'] = df['Occupation'].str.replace(r'[^a-zA-Z0-9\-]', '', regex=True)
    df['Education Level'] = df['Education Level'].str.replace('.', ' ', regex=False)


    df['Housing Loan'] = df['Housing Loan'].fillna('unknown')
    df['Personal Loan'] = df['Personal Loan'].fillna('unknown')

    for col in ['Housing Loan', 'Personal Loan','Contact Method']:
        df[col] = df[col].astype(str).str.lower()

    df['Contact Method'] = df['Contact Method'].replace({'cell': 'cellular'})
    df = df[df['Campaign Calls'] >= 0]
    
    df['Subscription Status'] = df['Subscription Status'].map({'yes': 1, 'no': 0})
    
    df.drop(columns=['Client ID'], inplace=True)
    
    education_years = {
    'illiterate': 0,
    'basic 4y': 4,
    'basic 6y': 6,
    'basic 9y': 9,
    'high school': 12,
    'professional course': 14,
    'university degree': 16,
    'unknown': -1  
    }
    df['Education Level'] = df['Education Level'].map(education_years)

    df['Had Previous Contact'] = (df['Previous Contact Days'] != 999).astype(int)
    df.drop(columns=['Previous Contact Days'], inplace=True)

    binary_cols = ['Credit Default', 'Housing Loan', 'Personal Loan']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0, 'unknown': -1})
    return df
#label encoder is use for random forest and XGBoost because they are tree-based models and can handle categorical values
# as distinct, without assuming any order. However, One-Hot Encoding can create more features (columns),
# which may lead to higher dimensionality and potential noise in certain cases so label encoder is use for random forest and XGBoost.
def encode_label(df):
    label_cols = ['Occupation', 'Marital Status', 'Education Level', 'Contact Method']
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df
# One Hot encoding is use for logistic Regression because  Label Encoding can mislead the logistic Regression model by 
# implying an artificial order between categories (e.g., Red < Blue < Green, Green is 2, Blue is 1 and Red is 0 )
# which doesn’t exist and The model doesn't know that 0 < 1 < 2 unless explicitly informed. 
# In many cases, it is incorrect to impose such an order, especially when categories are purely nominal 
# (i.e., they don't have an inherent ranking or order, such as colors or product names).
# One Hot encoding ensures distinct and independent that's why, it is use for logistic Regression
def encode_onehot(df):
    cat_cols = ['Occupation', 'Marital Status', 'Education Level', 'Contact Method']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_encoded = encoder.fit_transform(df[cat_cols])
    encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols))
    encoded_df.index = df.index
    df = df.drop(columns=cat_cols)
    df = pd.concat([df, encoded_df], axis=1)
    return df
#StandardScaler is used to adjust numerical features so they have a mean of 0 and a standard deviation of 1. 
# This ensures that no feature with a larger range (like  Age might range from 18 to 100, ) affects the model more than others (like Campaign Calls might range from 0 to 43).
# It's especially important for models like Logistic Regression and KNN, which are sensitive to the scale of the features.
def scale_numerical(df):
    scaler = StandardScaler()
    num_cols = ['Age', 'Campaign Calls']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df
# Step of the preprocessing on which should be use for each of the models
def preprocess_data(df, model_type='logreg'):
    df = basic_cleaning(df)
    df = scale_numerical(df)
    if model_type == 'logreg':
        df = encode_label(df)
    elif model_type in ['rf', 'xgb']:
        df = encode_onehot(df)
    return df