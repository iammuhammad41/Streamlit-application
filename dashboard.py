import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import joblib

# Load models
model = joblib.load("models/text_classifier.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

st.title("OSINT Weapon-Related Dashboard")

# Text input
text_input = st.text_area("Enter text to classify:")
if st.button("Classify"):
    text_vectorized = tfidf.transform([text_input])
    prediction = model.predict(text_vectorized)
    st.write("Prediction: Weapon-Related" if prediction[0] == 1 else "Prediction: Not Weapon-Related")

# Upload and visualize data
uploaded_file = st.file_uploader("Upload Text File", type="txt")
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    wordcloud = WordCloud(width=800, height=400).generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)
