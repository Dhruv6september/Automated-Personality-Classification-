import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from preprocess import preprocess

#ensuring model folder exists
os.makedirs("model", exist_ok=True)

#loading dataset
df = pd.read_csv("data/mbti_1.csv")

# Keep relevant columns
df = df[["type", "posts"]]

#introvert/extrovert
def convert_label(mbti):
    return "introvert" if mbti[0] == "I" else "extrovert"

df["label"] = df["type"].apply(convert_label)
df["text"] = df["posts"]

# dataset size
df = df.sample(8000, random_state=42)

# Preprocess
df["clean_text"] = df["text"].apply(preprocess)

#balancing dataset
df_ext = df[df["label"] == "extrovert"]
df_int = df[df["label"] == "introvert"]

df = pd.concat([
    df_ext.sample(len(df_int), replace=True),
    df_int
])

#TF-IDF with bigrams
vectorizer = TfidfVectorizer(
    max_features=7000,
    ngram_range=(1,2),
    min_df=2
)

X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

#train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#SVM
model = SVC(kernel='linear', probability=True)

#train
model.fit(X_train, y_train)

#evaluate
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("Accuracy:", accuracy)

#save model
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/tfidf.pkl")

print("Model saved successfully!")