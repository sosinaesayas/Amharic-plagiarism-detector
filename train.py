import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
import numpy as np
import os



def main():
    script_dir = os.path.dirname(__file__)
    features_file = os.path.join(script_dir, 'sample_statistical_features.csv')
    labels_file = os.path.join(script_dir, 'cleaned.csv')

    # Load features and labels
    features_df = pd.read_csv(features_file)
    labels_df = pd.read_csv(labels_file)

    # Adjust datasets to have the same number of rows
    min_length = min(len(features_df), len(labels_df))
    features_df = features_df.head(min_length)
    labels_df = labels_df.head(min_length)

    # Drop non-numeric columns (or handle them appropriately)
    features_df = features_df.select_dtypes(include=[np.number])

    # Impute NaN values in numeric columns
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    features_df = pd.DataFrame(imputer.fit_transform(features_df), columns=features_df.columns)

    # Check and handle NaN values in labels
    if labels_df['label'].isnull().any():
        labels_df['label'].fillna(0, inplace=True)  # Replace NaN with 0 or another appropriate value

    # Assign features and labels
    X = features_df
    y = labels_df['label']

    # Train-test split and model training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = LogisticRegression(C=0.00001, max_iter=10000000000)
    classifier.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = classifier.predict(X_test)
    print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
    print("Accuracy: {:.2f}".format(accuracy_score(y_test, y_pred)))
    print("Precision: {:.2f}".format(precision_score(y_test, y_pred, average='binary')))
    print("Recall: {:.2f}".format(recall_score(y_test, y_pred, average='binary')))
    print("F1 Score: {:.2f}".format(f1_score(y_test, y_pred, average='binary')))

if __name__ == "__main__":
    main()
