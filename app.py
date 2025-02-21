import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# Load NLTK stopwords
nltk.download('stopwords')

# Load the saved SVM model and vectorizer
with open("svm_fake_news_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Function to clean input text
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# Streamlit UI
st.title("📰 Fake News Detection System")
st.write("Enter a news article to check if it's **Fake** or **Real**")

# User input
user_input = st.text_area("Paste News Article Here", height=200)

if st.button("Analyze"):
    if user_input:
        cleaned_text = preprocess_text(user_input)  # Clean input
        transformed_text = vectorizer.transform([cleaned_text])  # Convert to vector
        prediction = model.predict(transformed_text)[0]  # Predict

        if prediction == 1:
            st.success("✅ This news is **Real**.")
        else:
            st.error("🚨 This news is **Fake**.")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
