import streamlit as st
import pickle
import pandas as pd

# Load Model and Vectorizer

with open("nb_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# Streamlit App UI

st.set_page_config(page_title="News Category Classifier", layout="centered")

st.title("News Category Classifier")
st.write("Enter a headline or upload a CSV file to classify news into **World**, **Sports**, **Business**, or **Sci/Tech** categories.")

# Category mapping
category_mapping = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

# Single Headline Prediction
user_input = st.text_area("Enter a news headline:")

if st.button("Predict Category"):
    if user_input.strip():
        text_tfidf = vectorizer.transform([user_input])
        pred = model.predict(text_tfidf)[0]
        st.success(f"Predicted Category: **{category_mapping[pred]}**")
    else:
        st.warning("Please enter a news headline before predicting!")

# CSV Upload for Batch Prediction
# ------------------------------
st.markdown("---")
st.subheader("📂 Upload a CSV file for Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV with a 'text' column", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "text" not in df.columns:
        st.error("❌ The uploaded CSV must contain a column named 'text'.")
    else:
        text_tfidf = vectorizer.transform(df["text"].astype(str))
        preds = model.predict(text_tfidf)
        df["Predicted_Label"] = preds
        df["Predicted_Category"] = df["Predicted_Label"].map(category_mapping)
        
        st.success("✅ Prediction Completed!")
        st.dataframe(df.head())

        # Download button
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name="predicted_news.csv",
            mime="text/csv"
        )
