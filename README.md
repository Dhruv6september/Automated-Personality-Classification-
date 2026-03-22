# Automated-Personality-Classification-

A simple Natural Language Processing (NLP) project that classifies text as **Introvert** or **Extrovert** using Machine Learning.

Built using **Python, Scikit-learn, NLTK, and Streamlit**, this project demonstrates an end-to-end ML pipeline from data preprocessing to real-time prediction.

---

## 🚀 Features

- 🔍 Text classification using **TF-IDF + SVM**
- 🧹 Text preprocessing with **NLTK**
  - Tokenization  
  - Stopword removal  
  - Normalization  
- 📊 Confidence score for predictions  
- 🎨 Interactive UI using **Streamlit**
- ⚡ Real-time predictions  

---

## 🛠️ Tech Stack

- Python  
- Scikit-learn  
- NLTK  
- Pandas  
- Streamlit  
- Joblib  

---

## 📂 Project Structure
#### personality-classifier/
#### │
#### ├── data/
#### │ └── mbti_1.csv
#### │
#### ├── model/
#### │ ├── model.pkl
#### │ └── tfidf.pkl
#### │
#### ├── preprocess.py
#### ├── train.py
#### ├── streamlit_app.py
#### │
#### ├── requirements.txt
#### └── README.md

---

## 🧠 How It Works

1. **Dataset**  
   Uses the MBTI dataset from Kaggle.

2. **Preprocessing**  
   - Converts text to lowercase  
   - Removes punctuation and stopwords  
   - Tokenizes text using NLTK  

3. **Feature Extraction**  
   - TF-IDF Vectorization  
   - Uses unigrams + bigrams  

4. **Model Training**  
   - Support Vector Machine (SVM)  
   - Trained on labeled introvert/extrovert data  

5. **Prediction**  
   - Takes user input  
   - Processes text  
   - Predicts personality with confidence score  

---

## ▶️ How to Run

### 1. Clone the repository

git clone https://github.com/

### 2. Install the dependecies

pip install -r requirements.txt

### 3. Train the model

python train.py

### 4. Run the app

streamlit run streamlit_app.py

## 🎯 Accuracy
  - Achieves approximately ~80% accuracy.
  - Performance may vary based on dataset sampling and preprocessing.

## ⚠️ Limitations
  - Based on text patterns, not actual psychological evaluation.
  - MBTI dataset is noisy and subjective.
  - Short or ambiguous inputs may reduce accuracy.

## 💡 Future Improvements.
  - Improve dataset quality.
  - Use advanced NLP models (BERT, LSTM).
  - Add explainability (feature importance).
  - Enhance UI with more visualizations.
