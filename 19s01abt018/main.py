Twitter: https://twitter.com/Oracle
Facebook: https://www.facebook.com/Oracle/
Instagram: https://www.instagram.com/oracle/
LinkedIn: https://www.linkedin.com/company/oracle/
Website: https://www.oracle.com/
# Import  libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# Load data
df = pd.read_csv('oracle_comments.csv')
# Remove URLs
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

df['clean_text'] = df['text'].apply(lambda x: remove_urls(x))
# Remove special characters
def remove_special_characters(text):
    special_char_pattern = re.compile(r'[^a-zA-z0-9\s]')
    return special_char_pattern.sub(r'', text)

df['clean_text'] = df['clean_text'].apply(lambda x: remove_special_characters(x))

df['clean_text'] = df['clean_text'].apply(lambda x: x.lower())
# Tokenize the text data
nltk.download('punkt')
df['tokens'] = df['clean_text'].apply(lambda x: word_tokenize(x))
# Remove stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

df['tokens'] = df['tokens'].apply(lambda x: remove_stopwords(x))
# Stem the data using PorterStemmer
porter = PorterStemmer()

def stem_tokens(tokens):
    return [porter.stem(word) for word in tokens]

df['stemmed_tokens'] = df['tokens'].apply(lambda x: stem_tokens(x))
print(df['stemmed_tokens'].head())
0    [know, oracl, thought, leader, industri, dont,...
1             [googl, cloud, oracl, cloud, versus, see]
2    [happi, celebr, second, oracl, employe, day, j...
3    [want, know, oracl, employe, valu, like, learn...
4    [
