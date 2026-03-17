import pandas as pd
import re
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# from fastapi import FastAPI

# Download latest version

df = pd.read_csv("datas/spam.csv", encoding="latin-1")
df = df[['v1','v2']]
df.columns = ['label','text']

# Clean the text data
stop_words = set(stopwords.words("english"))
def clean_text(text):
  text = text.lower()
  text = re.sub(r'[^\w\s]', '', text)
  words = text.split()
  words = [w for w in words if w not in stop_words]
  return " ".join(words)
df["clean_text"] = df["text"].apply(clean_text)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_text"])

# Encode the labels
encoder = LabelEncoder()
y = encoder.fit_transform(df["label"])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)


pred = model.predict(X_test)
# print(accuracy_score(y_test, pred))
# print(classification_report(y_test, pred)) 

pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

def predict_spam(message):
  message = clean_text(message)
  vector = vectorizer.transform([message])
  prediction = model.predict(vector)
  return prediction
print(predict_spam("Play bet8x free"))  # Example usage

# app = FastAPI()

# model = pickle.load(open("spam_model.pkl","rb"))
# vectorizer = pickle.load(open("vectorizer.pkl","rb"))

# @app.get("/predict")
# def predict(text:str):
#   vector = vectorizer.transform([text])
#   result = model.predict(vector)
#   return {"spam": int(result[0])}