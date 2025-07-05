import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Load the fine-tuned model and tokenizer
@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("./model")
    tokenizer = DistilBertTokenizerFast.from_pretrained("./model")
    return model, tokenizer

model, tokenizer = load_model()

# App title and instructions
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector Chatbot")
st.markdown("Enter a news article or snippet, and I’ll tell you if it’s **Fake** or **True**, with confidence!")

# User input
news_input = st.text_area("Paste your news snippet here 👇", height=200)

# Prediction logic
if st.button("Detect"):
    if news_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            # Tokenize input
            inputs = tokenizer(news_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                pred = torch.argmax(probs).item()
                confidence = torch.max(probs).item() * 100

            # Output
            if pred == 0:
                st.error(f"🔴 This news is likely **FAKE** with {confidence:.2f}% confidence.")
            else:
                st.success(f"🟢 This news is likely **TRUE** with {confidence:.2f}% confidence.")

