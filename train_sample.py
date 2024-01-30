import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

def main():
    script_dir = os.path.dirname(__file__)
    csv_file = os.path.join(script_dir, 'sample_statistical_features.csv')

    # Load the dataset
    df = pd.read_csv(csv_file)

    # Drop rows with NaN values
    df = df.dropna()

    # Ensure there are enough samples after dropping NaN values
    if df.shape[0] < 2:
        print("Not enough data for training and testing after dropping NaN values.")
        return

    # Define features (X) and label (y)
    X = df.drop('label', axis=1)
    y = df['label']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the Logistic Regression model
    classifier = LogisticRegression(C=0.00001, max_iter=10000000000)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Calculate and print evaluation metrics
    print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
    print("Accuracy: {:.2f}".format(accuracy_score(y_test, y_pred)))
    print("Precision: {:.2f}".format(precision_score(y_test, y_pred)))
    print("Recall: {:.2f}".format(recall_score(y_test, y_pred)))
    print("F1 Score: {:.2f}".format(f1_score(y_test, y_pred)))

if __name__ == "__main__":
    main()
