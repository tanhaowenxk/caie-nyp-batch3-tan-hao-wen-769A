# Training for the models
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split, GridSearchCV


# Model Parameterization: allows to experiment with different hyperparameters without changing the script.
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path) as f:
        return json.load(f)

def train_model(df, model_type='logreg'):
    config = load_config()
    hyperparameters = config[model_type]
    # Get the features and the target
    X = df.drop(columns=['Subscription Status'])
    y = df['Subscription Status']
    #Spilt the data to x and y and strafity is to keep class distribution consistent between training and test
    #Random state is to be able to keep the result same every time the code is run.
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    #SMOTETomek is to balance the classes , it oversample the minority class which is the yes(1)
    # and remove bordeline majority class samples.
    smote_tomek = SMOTETomek(random_state=42)
    X_train_sm, y_train_sm = smote_tomek.fit_resample(X_train, y_train)

    if model_type == 'logreg':
        original_model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == 'rf':
        original_model = RandomForestClassifier(random_state=42)
    elif model_type == 'xgb':
        original_model = XGBClassifier(random_state=42, eval_metric='logloss')
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    grid_search = GridSearchCV(original_model, hyperparameters, scoring='f1', cv=5, n_jobs=-1)
    grid_search.fit(X_train_sm, y_train_sm)

    best_model = grid_search.best_estimator_

    # Train the original model manually
    original_model.fit(X_train_sm, y_train_sm)

    return original_model, best_model, X_test, y_test, grid_search.best_params_
