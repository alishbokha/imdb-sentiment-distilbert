import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Hugging Face model
MODEL_NAME = "alishbokha/imdb-distilbert"


# Load tokenizer and model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


tokenizer, model, device = load_model()


# Prediction function
def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)

    prediction = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = probabilities[0][prediction].item()

    sentiment = "Positive" if prediction == 1 else "Negative"

    return sentiment, confidence


# Streamlit UI
st.title("🎬 IMDb Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review and the fine-tuned "
    "DistilBERT model will predict its sentiment."
)

review = st.text_area(
    "Enter your movie review:",
    placeholder="Example: This movie was absolutely amazing!"
)

if st.button("Predict Sentiment"):
    if review.strip():
        sentiment, confidence = predict_sentiment(review)

        st.subheader("Prediction")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")
    else:
        st.warning("Please enter a movie review.")
