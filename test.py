import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.utils import class_weight
import numpy as np
import os
import joblib  # for loading the model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def main():
    # Path to the saved model
    model_path = os.path.join(os.path.dirname(__file__), "logistic_regression_model.pkl")

    test_csv_file = os.path.join(os.path.dirname(__file__), "statistical_features.csv")
    

    print("test_csv_file:", test_csv_file)


    test_df = pd.read_csv(test_csv_file)

    # Normalize labels
    normalize_label = lambda x: 1 if 'original' in str(x).lower().strip() else 0

    

    # Feature selection
    feature_columns = [str(i) for i in range(768)]

    X_test = test_df[feature_columns]
    # y_test = test_df['label']

    model = joblib.load(model_path)

        # Adjusting the decision threshold
    probabilities = model.predict_proba(X_test)

    print("probabilities:", probabilities)
    new_threshold = 0.5 # Example threshold, adjust based on your needs


    new_predictions = np.where(probabilities[:, 1] >= new_threshold, 1, 0)
    labels = ['plagiarized' if pred == 1 else 'original' for pred in new_predictions]

    # Print predictions with their respective probabilities
    for label, probability in zip(labels, probabilities[:, 1]):
        print(f'Prediction: {label}, Probability: {probability:.2f}')

    # print("New Confusion Matrix:", confusion_matrix(y_test, new_predictions))
    # print("New Accuracy:", accuracy_score(y_test, new_predictions))
    # print("New Precision:", precision_score(y_test, new_predictions))
    # print("New Recall:", recall_score(y_test, new_predictions))
    # print("New F1 Score:", f1_score(y_test, new_predictions))
if __name__ == "__main__":
    main()
