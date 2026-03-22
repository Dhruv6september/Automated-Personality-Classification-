import streamlit as st
import joblib
from preprocess import preprocess

#Load model
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/tfidf.pkl")

#Page config
st.set_page_config(page_title="Personality Classifier", page_icon="🧠")

#Title
st.title("🧠 Personality Classifier")
st.caption("Introvert vs Extrovert using NLP")

#Input
user_input = st.text_area("✍️ Enter text:")

#Word count
if user_input:
    st.caption(f"📝 Words: {len(user_input.split())}")

if st.button("🚀 Predict"):
    if user_input.strip() != "":
        
        with st.spinner("Analyzing..."):
            cleaned = preprocess(user_input)
            vector = vectorizer.transform([cleaned])

            # Prediction
            prediction = model.predict(vector)[0]

            #PROBABILITY MAPPING
            probs = model.predict_proba(vector)[0]
            classes = model.classes_

            intro_prob = probs[list(classes).index("introvert")] * 100
            extro_prob = probs[list(classes).index("extrovert")] * 100

        # Result
        st.subheader("🧾 Result")

        if intro_prob > extro_prob:
            st.success("🧘 Introvert")
        else:
            st.success("🎉 Extrovert")

        # Confidence
        st.subheader("📊 Confidence")
        st.progress(int(max(intro_prob, extro_prob)))

        col1, col2 = st.columns(2)
        col1.metric("🧘 Introvert", f"{intro_prob:.2f}%")
        col2.metric("🎉 Extrovert", f"{extro_prob:.2f}%")

        # Cleaned text
        with st.expander("🔍 See cleaned text"):
            st.code(cleaned)

    else:
        st.warning("⚠️ Please enter some text!")

# Sidebar
st.sidebar.title("About")
st.sidebar.info("TF-IDF + SVM + NLTK based NLP project")