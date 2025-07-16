# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import re
import string
import joblib
import os

# Hàm tiền xử lý
def preprocess_text(text):
    if isinstance(text, str):
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    else:
        return ""

# Step 1: Load dataset
df = pd.read_csv("./data/SpamCollectionSMS.txt", sep='\t', header=None, names=['label', 'message'])

# Step 2: Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Xoá dòng bị thiếu dữ liệu
df = df.dropna(subset=['message'])

# Ép kiểu về chuỗi để tránh lỗi .lower()
df['message'] = df['message'].astype(str)

# Apply preprocessing
df['message'] = df['message'].apply(preprocess_text)

# Step 4: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Step 5: Vectorize using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Huấn luyện mô hình với thuật toán Naive Bayes
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 7: Predict and evaluate
predictions = model.predict(X_test_vec)

print("📊 Accuracy:", accuracy_score(y_test, predictions))
print("\n🔍 Classification Report:\n", classification_report(y_test, predictions))

joblib.dump(model, "spam_classifier_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
