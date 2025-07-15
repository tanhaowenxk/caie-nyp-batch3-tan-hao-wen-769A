# Main acts as the control centre for the whole pipeline 
import sys
from load_data import load_data
from preprocess import preprocess_data
from train_model import train_model
from evaluate import evaluate_model

def main():
    # Check whether the user has provided the model type correctly, or else it will exit
    if len(sys.argv) < 2:
        print("Please provide a model type: logreg, rf, or xgb")
        sys.exit(1)

    model_type = sys.argv[1]

    print(f"Starting pipeline for model: {model_type}")
    
    df = load_data()
    df_clean = preprocess_data(df, model_type=model_type)
    
    # Train both the original model and the best model  GridSearchCV)
    original_model, best_model, X_test, y_test, best_params  = train_model(df_clean, model_type=model_type)
    
    evaluate_model(
        original_model,
        best_model,
        X_test,
        y_test,
        best_params=best_params,
        model_type=model_type
    )

    print(f"Pipeline completed for model: {model_type}")

if __name__ == "__main__":
    main()
