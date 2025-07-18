# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import re
import string
import joblib

# Function to preprocess text data
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

# Step 2: Preprocessing data
# Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
# remove rows with missing messages
df = df.dropna(subset=['message'])
# Ensure 'message' column is string type
df['message'] = df['message'].astype(str)
# Apply preprocessing
df['message'] = df['message'].apply(preprocess_text)

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Step 4: Vectorize using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train model with Naive Bayes agorithm
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 6: Predict and evaluate
predictions = model.predict(X_test_vec)

print("📊 Accuracy:", accuracy_score(y_test, predictions))
print("\n🔍 Classification Report:\n", classification_report(y_test, predictions))

# Step 7: Save the model and vectorizer to reuse later
joblib.dump(model, "spam_classifier_model.pkl")
joblib.dump(vectorizer, "spam_tfidf_vectorizer.pkl")