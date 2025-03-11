import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load dataset (assuming 'UpdatedResumeDataSet.csv' exists)
df = pd.read_csv("UpdatedResumeDataSet.csv")

# Extract text and labels
texts = df["Resume"].astype(str)  # Resume text
labels = df["Category"]  # Job category

# Convert labels to numerical values
le = LabelEncoder()
y = le.fit_transform(labels)

# Convert text to TF-IDF features
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(texts)

# Train an SVM model
svc_model = SVC(probability=True)
svc_model.fit(X, y)

# Save the trained model, vectorizer, and label encoder
pickle.dump(svc_model, open("clf.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))
pickle.dump(le, open("encoder.pkl", "wb"))

print("✅ Model training complete! Saved clf.pkl, tfidf.pkl, and encoder.pkl.")


