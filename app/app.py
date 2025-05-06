import streamlit as st
from transformers import AutoTokenizer,TFAutoModelForSequenceClassification
import tensorflow as tf
import os

st.title("Sentiment Analysis using Distilbert")
st.subheader("Analyze the sentiment of you text using the Distilbert model")

def load_model():
    # Get absolute path to the folder containing this script (app.py)
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Construct paths relative to app.py
    tokenizer_path = os.path.join(base_path, 'Tokenizer')
    model_path = os.path.join(base_path, 'Model')

    # Load tokenizer and model from these folders
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    mod = TFAutoModelForSequenceClassification.from_pretrained(model_path)

    return tok, mod
tokenizer,model=load_model()

def predict_sentiment(text):
    input = tokenizer(text, return_tensors='tf', padding=True, truncation=True, max_length=128)
    output = model(input_ids=input['input_ids'], attention_mask=input['attention_mask'])
    logits = output.logits

    probs = tf.nn.softmax(logits, axis=1)  

    predicted_class = tf.argmax(probs, axis=1).numpy()[0]  
    confidence = probs[0][predicted_class].numpy() 
    
    label_map = {0: "Negative", 1: "Positive"}  

   
    label = label_map.get(predicted_class, "Unknown")  # Default to "Unknown" if no match
    return label, confidence


user_input = st.text_area("Enter your text below:", height=150)

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            label, confidence = predict_sentiment(user_input)
            
            # ---- Display result ----
            st.markdown(f"**Prediction:** `{label}`")
            st.markdown(f"**Confidence:** `{confidence*100:.2f}%`")






    