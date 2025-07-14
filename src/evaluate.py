# Evaluation for the model
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import numpy as np

def evaluate_model(original_model, best_model, X_test, y_test):
    # For the original model get the best threshold for the highest F1 score
    print("Evaluating Original Model...")
    y_probs_original = original_model.predict_proba(X_test)[:, 1]
    precision_original, recall_original, thresholds_original = precision_recall_curve(y_test, y_probs_original)
    f1_scores_original = 2 * (precision_original * recall_original) / (precision_original + recall_original + 1e-6)
    best_thresh_idx_original = np.argmax(f1_scores_original)
    best_threshold_original = thresholds_original[best_thresh_idx_original]

    print(f"Original Model - Best Threshold: {best_threshold_original:.4f}")
    print(f"Original Model - Best F1 Score: {f1_scores_original[best_thresh_idx_original]:.4f}")

    # Apply the best threshold for the original model
    y_pred_original = (y_probs_original >= best_threshold_original).astype(int)

    # Print classification report and ROC AUC for the original model
    print("Original Model Classification Report:")
    print(classification_report(y_test, y_pred_original, digits=4))
    print("Original Model ROC AUC Score:", roc_auc_score(y_test, y_probs_original))

    # For the best model for GridSearchCV and get the best threshold for the highest F1 score
    print("\nEvaluating Best Model from GridSearchCV...")
    y_probs_best = best_model.predict_proba(X_test)[:, 1]
    precision_best, recall_best, thresholds_best = precision_recall_curve(y_test, y_probs_best)
    f1_scores_best = 2 * (precision_best * recall_best) / (precision_best + recall_best + 1e-6)
    best_thresh_idx_best = np.argmax(f1_scores_best)
    best_threshold_best = thresholds_best[best_thresh_idx_best]

    print(f"Best Model - Best Threshold: {best_threshold_best:.4f}")
    print(f"Best Model - Best F1 Score: {f1_scores_best[best_thresh_idx_best]:.4f}")

    # Apply the best threshold for the best model
    y_pred_best = (y_probs_best >= best_threshold_best).astype(int)

    # Print classification report and ROC AUC for the best model
    print("Best Model Classification Report:")
    print(classification_report(y_test, y_pred_best, digits=4))
    print("Best Model ROC AUC Score:", roc_auc_score(y_test, y_probs_best))

    return best_threshold_original, best_threshold_best
