import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import torch
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def load_bbc_data(data_dir="data/bbc", max_features=100):
    texts, labels = [], []
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):
            for fname in os.listdir(class_dir):
                with open(os.path.join(class_dir, fname), 'r', encoding='latin1') as f:
                    texts.append(f.read())
                    labels.append(label)

    def preprocess(text):
      tokens = text.lower().split()
      tokens = [t for t in tokens if t.isalpha() and t not in stopwords.words('english')]
      return ' '.join(tokens)

    processed_texts = [preprocess(t) for t in texts]

    # View 1: TF-IDF
    vec1 = TfidfVectorizer(max_features=max_features)
    vec2 = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))

    x1 = vec1.fit_transform(processed_texts).toarray()
    x2 = vec2.fit_transform(processed_texts).toarray()

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    return torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
