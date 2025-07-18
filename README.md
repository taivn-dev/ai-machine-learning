# Machine Learning Demo

## 📁 Project Structure

```
ai-machine-learning/
│
├── backend/
│   ├── app/
│   │   ├── supervised_learning.py
│   │   ├── unsupervised_learning.py
│   │   ├── reinforcement_learning.py
│   │   ├── main.py
│   │   └── data/
│   │       └── mall_customers.csv
│   │       └── SpamCollectionSMS.txt
│   ├── venv/     # Virtual environment (not tracked by git)
│   └── requirements.txt
│
├── README.md
└── .gitignore
```

## 🛠️ Environment Setup

1. Install Python
   https://www.python.org/downloads/. After that, check python is successfully installed by run the command:
   ```bash
   python --version

2. Clone the repository:
   ```
   git clone https://github.com/taivn-dev/ai-machine-learning.git
   cd ai-machine-learning
   ```

3. Create a virtual environment:
   ```
   cd backend
   python -m venv venv
   ```

4. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

5. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

## Run Demo
1. Supervised Learning:
   ```
   cd app
   python supervised_learning.py
   ```
   This will create model and vector for supervised learning

   To start the FastAPI application, run the following command:
   ```
   uvicorn app.main:app --reload
   ```
   This will start the server with auto-reload enabled for development.

   ### Usage
   Once the server is running, you can access the API documentation at:
   ```
   http://127.0.0.1:8000/docs
   ```

2. Unsupervised Learning:
   ```
   cd app
   python unsupervised_learning.py
   ```

3. Reinforcement Learning:
   ```
   cd app
   python reinforcement_learning.py
   ```