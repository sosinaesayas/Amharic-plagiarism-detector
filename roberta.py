import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import os
import joblib  # For saving the model
from normalizer import normalizer
from lemmatizer import lemmatize_text
from stopword_remover import remove_stopwords
# Define your preprocessing functions
def tokenize(text):
    return text.split()

def process_text(text):
    
    normalized_text = normalizer(text)
    tokens = tokenize(normalized_text)
    lemmatized_tokens = [lemmatize_text(token) for token in tokens]
    lemmatized_text = ' '.join(lemmatized_tokens)
    stopwords_removed_text = remove_stopwords(lemmatized_text)

    print('preprocessing finished')
    return stopwords_removed_text

# Function to extract features
def extract_features(text):
    print('extracting features...')
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    print('features extracted')
    return outputs.logits.mean(dim=1).squeeze().detach().numpy().flatten()

# Load and preprocess training data
script_dir = os.path.dirname(__file__)
train_file = os.path.join(script_dir, 'cleaned.csv')
df_train = pd.read_csv(train_file)
df_train['PostContent'] = df_train['PostContent'].apply(process_text)

# Load and preprocess test data
test_file = os.path.join(script_dir, 'test.csv')
df_test = pd.read_csv(test_file)
df_test['PostContent'] = df_test['PostContent'].apply(process_text)
preprocessed_test_file = os.path.join(script_dir, 'preprocessed_test_data.csv')
df_test.to_csv(preprocessed_test_file, index=False)

# Initialize RoBERTa model and tokenizer
model_path = "C:\\Users\\hp\\.cache\\huggingface\\transformers\\xlm-roberta-base-finetuned-ner-amharic"
tokenizer = AutoTokenizer.from_pretrained("mbeukman/xlm-roberta-base-finetuned-ner-amharic")
model = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True)

# Feature extraction
X_train = np.array(df_train['PostContent'].apply(extract_features).tolist())
y_train = df_train['label'].values

# Train the logistic regression model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Save the model
model_save_path = os.path.join(script_dir, 'logistic_regression_model.pkl')
joblib.dump(classifier, model_save_path)

# Load preprocessed test data
df_test_preprocessed = pd.read_csv(preprocessed_test_file)
X_test = np.array(df_test_preprocessed['PostContent'].apply(extract_features).tolist())
y_test = df_test_preprocessed['label'].values

# Test the model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print results
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
