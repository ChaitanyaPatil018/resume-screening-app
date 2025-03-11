# Required Installations:
# pip install streamlit scikit-learn python-docx PyPDF2 joblib

import streamlit as st
import pickle
import joblib  # For loading models safely
import docx  # Extract text from Word file
import PyPDF2  # Extract text from PDF
import os
import re

# ------------------------------
# ✅ Function to check if a file exists
# ------------------------------
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        st.error(f"❌ Error: The required file `{file_path}` is missing! Please ensure it exists in the project directory.")
        st.stop()

# ------------------------------
# ✅ Load Pre-trained Model and TF-IDF Vectorizer
# ------------------------------
# Ensure required files exist
check_file_exists("clf.pkl")
check_file_exists("tfidf.pkl")
check_file_exists("encoder.pkl")

# Try loading with pickle, fallback to joblib if needed
try:
    svc_model = pickle.load(open("clf.pkl", "rb"))
except Exception as e:
    st.error(f"❌ Error loading `clf.pkl`: {e}")
    st.stop()

try:
    tfidf = joblib.load("tfidf.pkl")  # Use joblib in case it was saved with joblib
except Exception:
    try:
        tfidf = pickle.load(open("tfidf.pkl", "rb"))  # Fallback to pickle
    except Exception as e:
        st.error(f"❌ Error loading `tfidf.pkl`: {e}")
        st.stop()

try:
    le = pickle.load(open("encoder.pkl", "rb"))
except Exception as e:
    st.error(f"❌ Error loading `encoder.pkl`: {e}")
    st.stop()

# ------------------------------
# ✅ Function to Clean Resume Text
# ------------------------------
def clean_resume(text):
    text = re.sub(r"http\S+\s", " ", text)  # Remove URLs
    text = re.sub(r"RT|cc", " ", text)  # Remove retweets and CC
    text = re.sub(r"#\S+\s", " ", text)  # Remove hashtags
    text = re.sub(r"@\S+", " ", text)  # Remove mentions
    text = re.sub(r"[!\"#$%&'()*+,-./:;<=>?@\[\\\]^_`{|}~]", " ", text)  # ✅ Corrected
  # ✅ Fixed

    text = re.sub(r"[^\x00-\x7f]", " ", text)  # Remove non-ASCII characters
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    return text.strip()

# ------------------------------
# ✅ Functions to Extract Text from Files
# ------------------------------
def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        return text if text.strip() else "⚠️ No readable text found in PDF!"
    except Exception as e:
        return f"❌ Error extracting text from PDF: {e}"

def extract_text_from_docx(file):
    """Extract text from a DOCX file."""
    try:
        doc = docx.Document(file)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    except Exception as e:
        return f"❌ Error extracting text from DOCX: {e}"

def extract_text_from_txt(file):
    """Extract text from a TXT file with encoding handling."""
    try:
        return file.read().decode("utf-8")
    except UnicodeDecodeError:
        try:
            return file.read().decode("latin-1")  # Fallback encoding
        except Exception as e:
            return f"❌ Error reading TXT file: {e}"

# ------------------------------
# ✅ Function to Handle File Upload
# ------------------------------
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split(".")[-1].lower()

    if file_extension == "pdf":
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == "docx":
        return extract_text_from_docx(uploaded_file)
    elif file_extension == "txt":
        return extract_text_from_txt(uploaded_file)
    else:
        st.error("❌ Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
        st.stop()

# ------------------------------
# ✅ Function to Predict Resume Category
# ------------------------------
def predict_resume_category(resume_text):
    """Predicts the job category from the extracted resume text."""
    cleaned_text = clean_resume(resume_text)

    try:
        # Vectorize the cleaned text using the same TF-IDF vectorizer
        vectorized_text = tfidf.transform([cleaned_text])
        
        # Convert sparse matrix to dense (if required)
        vectorized_text = vectorized_text.toarray()

        # Predict category
        predicted_category = svc_model.predict(vectorized_text)

        # Convert numeric label back to category name
        predicted_category_name = le.inverse_transform(predicted_category)

        return predicted_category_name[0]  # Return the predicted category
    except Exception as e:
        return f"❌ Error in prediction: {e}"

# ------------------------------
# ✅ Streamlit UI Layout
# ------------------------------
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")

    st.title("📄 Resume Category Prediction App")
    st.markdown("Upload a resume (PDF, TXT, DOCX) to predict the job category using AI.")

    uploaded_file = st.file_uploader("📂 Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        st.write("✅ File uploaded successfully!")

        # Extract text from uploaded file
        try:
            resume_text = handle_file_upload(uploaded_file)
            
            if "❌" in resume_text or "⚠️" in resume_text:
                st.error(resume_text)
                return

            st.write("✅ Successfully extracted text from the uploaded resume.")

            # Display extracted text (optional)
            if st.checkbox("📝 Show extracted text", False):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            # Predict category
            st.subheader("📌 Predicted Job Category")
            category = predict_resume_category(resume_text)

            if "❌" in category:
                st.error(category)
            else:
                st.success(f"🎯 The predicted category is: **{category}**")

        except Exception as e:
            st.error(f"❌ Error processing the file: {str(e)}")

if __name__ == "__main__":
    main()
