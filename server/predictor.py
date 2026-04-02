import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import os

try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        pass

from nltk.corpus import stopwords
import contractions
stop_words = stopwords.words('english')
from nltk import WordNetLemmatizer

def process_text(df):
  final_abstracts = []
  lem = WordNetLemmatizer()
  for i in range(len(df)):
    l = [sent for sent in nltk.sent_tokenize(df['Abstract'][i])]
    l_words = []
    l_new = {}
    sentences = []
    for i in range(len(l)):
      words = nltk.word_tokenize(l[i])

      #fix punctuations
      sentence = ' '.join(words).replace(' , ',', ').replace(' .','.').replace(' !','!')
      sentence = sentence.replace(' ?','?').replace(' : ',': ').replace(' \'', '\'')
      sentence = sentence.replace(' ’ ', '\'').replace(' ’', '\'').replace('\n', '')

      #expand contractions
      expanded_words = []
      for word in sentence.split():
        expanded_words.append(contractions.fix(word))
      expanded_sent = ' '.join(expanded_words)
      words = nltk.word_tokenize(expanded_sent)
      l_new[sentence] = words
      sentences.append(sentence)

    #remove stopwords and punctuations
    for sentence in l_new.keys():
      new_words = []
      for word in l_new[sentence]:
        if word not in '!?.,;:' and word not in stop_words:
          new_words.append(word)
      l_new[sentence] = new_words

    #lemmatize the extracted words
    processed_data = {}
    for sent in l_new.keys():
      new_words = []
      for word in l_new[sent]:
        new_words.append(lem.lemmatize(word))
      processed_data[sent] = new_words

    #rejoin the sentences
    lst = []
    for sent in processed_data.keys():
      lst.append(' '.join(processed_data[sent]))
    final_abstracts.append(' '.join(lst))

  #convert the extracted features to dataframe
  processed_features = pd.DataFrame(final_abstracts, columns=['Abstract'])
  return processed_features

def predict(text: str):
    train_df = pd.read_csv('dataset_model/train_features.csv')
    test_df = pd.DataFrame([text], columns=['Abstract'])
    test_df = process_text(test_df)
    vectorizer = TfidfVectorizer()
    vectorizer.fit(train_df['Abstract'])
    vectors = vectorizer.transform(test_df['Abstract']).todense()

    filename = 'dataset_model/model.pkl'
    with open(filename, 'rb') as f:
        x = pickle.load(f)
        return x.predict(vectors).todense()