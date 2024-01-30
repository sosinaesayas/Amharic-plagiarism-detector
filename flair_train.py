import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from flair.data import Sentence
from flair.models import TextClassifier
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.trainers import ModelTrainer
from flair.datasets import FlairDatapointDataset
import os
from flair.data import Dictionary  # Import Dictionary
from flair.data import Corpus

script_dir = os.path.dirname(__file__)
csv_file = os.path.join(script_dir, 'cleaned.csv')
df = pd.read_csv(csv_file)

df['PostContent'] = df['PostContent'].fillna('')
df['PostContent'] = df['PostContent'].astype(str)

num_iterations = 8

print("Initializing embeddings...")
word_embeddings = [WordEmbeddings('glove'), FlairEmbeddings('multi-forward'), FlairEmbeddings('multi-backward')]
document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=512, reproject_words=True, reproject_words_dimension=256)

for i in range(num_iterations):
    print("Iteration:", i + 1)
    train_df, test_df = train_test_split(df, test_size=0.2)

    train_sentences = [Sentence(row['PostContent']).add_label('label', row['label']) for _, row in train_df.iterrows()]
    test_sentences = [Sentence(row['PostContent']).add_label('label', row['label']) for _, row in test_df.iterrows()]

    train_data = FlairDatapointDataset(train_sentences)
    test_data = FlairDatapointDataset(test_sentences)

   
    # Manually create label dictionary
    label_dict = Dictionary(add_unk=False)

    for i in range(num_iterations):
        # Split the dataset
        print("Iteration : ", i+1)
        train_df, test_df = train_test_split(df, test_size=0.2)

        # Convert dataframes to lists of Sentences
        train_sentences = [Sentence(row['PostContent']).add_label('label', str(row['label'])) for index, row in train_df.iterrows()]
        test_sentences = [Sentence(row['PostContent']).add_label('label', str(row['label'])) for index, row in test_df.iterrows()]

        # Create a Corpus object with the lists of Sentences
        corpus = Corpus(train=train_sentences, test=test_sentences, dev=None)

        # Create the text classifier with the corpus's label dictionary
        label_dict = corpus.make_label_dictionary('label')
        classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, label_type='label')

        # Train the model
        trainer = ModelTrainer(classifier, corpus)
        trainer.train(script_dir, max_epochs=10)

        # Rest of your evaluation code...

# Replace with your desired output directory



    # Evaluate the model
    model = TextClassifier.load(os.path.join(script_dir, 'final-model.pt'))
    predictions, true_labels = [], []
    for sentence in test_sentences:
        model.predict(sentence)
        predictions.append(sentence.labels[0].value)
        true_labels.append(sentence.get_label('label').value)

    accuracy = accuracy_score(true_labels, predictions)
    conf_matrix = confusion_matrix(true_labels, predictions)

    print(f"Iteration {i+1}:")
    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\n")
