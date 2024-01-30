import pandas as pd
from flair.data import Sentence
from flair.models import TextClassifier
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.trainers import ModelTrainer
from flair.datasets import Corpus

# Load dataset
df = pd.read_csv('cleaned.csv')

# Preprocess if necessary (ensure no NaN values in 'PostContent' and 'label')
df.dropna(subset=['PostContent', 'label'], inplace=True)

# Initialize embeddings
word_embeddings = [WordEmbeddings('glove'), FlairEmbeddings('multi-forward'), FlairEmbeddings('multi-backward')]
document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=512, reproject_words=True, reproject_words_dimension=256)

# Convert dataframe to list of Sentences
sentences = [Sentence(row['PostContent']).add_label('label', str(row['label'])) for index, row in df.iterrows()]

# Create a Corpus object
corpus = Corpus(train=sentences, test=None, dev=None)

# Create the text classifier
label_dict = corpus.make_label_dictionary('label')
classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, label_type='label')

# Train the model
trainer = ModelTrainer(classifier, corpus)
trainer.train('output', max_epochs=10)  # Replace 'output' with your desired output directory

# Save the model
model_path = 'output/final-model.pt'  # Replace 'output/final-model.pt' with your actual model path
classifier = TextClassifier.load(model_path)
classifier.save('plagiarism_detector_model.pkl')  # Saving as .pkl for convenience
