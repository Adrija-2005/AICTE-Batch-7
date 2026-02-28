import streamlit as st
import joblib

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("model.jb")

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection System")
st.write("AI system to detect whether news is Fake or Real.")

text = st.text_area("Enter News Article:")

if st.button("Analyze"):
    if text.strip():
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)
        probability = model.predict_proba(vec)

        confidence = max(probability[0]) * 100

        if prediction[0] == 1:
            st.success("🟢 Likely Real News")
        else:
            st.error("🔴 Likely Fake News")

        st.info(f"Confidence: {confidence:.2f}%")
    else:
        st.warning("Please enter some text.")