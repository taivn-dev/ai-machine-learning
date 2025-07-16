from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import string

# Load mô hình và vectorizer đã huấn luyện
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Khởi tạo FastAPI
app = FastAPI()

# Schema cho request
class Message(BaseModel):
    text: str

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

@app.get("/")
def index():
    return {"message" : "Welcome to Machine Learning Demo"}

# Endpoint kiểm tra spam
@app.post("/predict")
def predict_spam(msg: Message):
    clean_text = preprocess_text(msg.text)
    
    vector = vectorizer.transform([clean_text])
    prediction = model.predict(vector)[0]
    result = "spam" if prediction == 1 else "not spam"
    return {"result": result}
