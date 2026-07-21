from sklearn.metrics import (accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    """
    Liczy standardowy zestaw metryk klasyfikacji binarnej dla wytrenowanego modelu
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics, y_pred, y_pred_proba

def print_metrics(metrics, model_name="Model"):
    """Wypisuje metryki"""
    print(f"--- {model_name} ---")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1=score: {metrics['f1']:.3f}")
    print(f"ROC_AUC: {metrics['roc_auc']:.3f}")

def plot_confusion_matrix(y_test, y_pred, model_name="Model"):
    """Rysuje macierz pomyłek dla danych predykcji"""
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Spłacił','Nie spłacił'])
    disp.plot(cmap='Blues')
    plt.title(f'Macierz pomyłek - {model_name}')
    plt.show()
    