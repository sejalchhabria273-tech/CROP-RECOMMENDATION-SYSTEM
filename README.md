🌱 Crop Recommendation System (Machine Learning Project)
📌 Project Overview
This project aims to recommend the most suitable crop for farming based on soil nutrients and climate conditions using Machine Learning techniques. It demonstrates a complete ML pipeline from data preprocessing to deployment with Streamlit.

🎯 Objective
To build an end-to-end ML system that:
Takes farm data (soil and climate features) as input
Predicts the best crop to grow
Provides real-time recommendations via a web interface
Suggests farming tips to improve soil health and yield

📊 Features Used
Nitrogen (N)
Phosphorus (P)
Potassium (K
Temperature (°C)
Humidity (%)
Soil pH
Rainfall (mm)

🧠 Machine Learning Models Used
Decision Tre
Random Forest ✅ (Best Performing Model)
Logistic Regression

📈 Evaluation Metrics
Accuracy
Precision
Recall
F1 Score

🏆 Final Model
The Random Forest Classifier was selected as the final model because it achieved:

Highest accuracy
Robust performance across different crop categories

🛠️ Tech Stack
Python
Pandas
NumPy
Scikit-learn
Joblib
Streamlit

🚀 How to Run the Project Locally
1️⃣ Clone the Repository

bash
git clone https://github.com/your-username/Crop-Recommendation-System.git
cd Crop-Recommendation-System
2️⃣ Install Dependencies

bash
pip install -r requirements.txt
3️⃣ Run the Streamlit App

bash
streamlit run app.py
4️⃣ Open in Browser
The app will open automatically at:

Code
http://localhost:8501
🌐 Deployment
The project is deployed using Streamlit Cloud.

👉 Live Demo: Network URL: http://10.102.144.248:8501

📂 Project Structure
Code
├── app.py
├── models/
│   ├── crop_model.pkl
│   └── label_encoder.pkl
├── requirements.txt
└── README.md
💡 Key Highlights
Cleaned and processed agricultural dataset

Feature engineering for better accuracy

Compared multiple ML models

Built interactive UI using Streamlit

Provides crop prediction + farming suggestions

Deployed as a web application

🎤 Conclusion
This project demonstrates how Machine Learning can assist farmers by recommending suitable crops based on soil and climate conditions. It provides a scalable foundation for smart agriculture solutions.

👨‍💻 Author
SEJAL

⭐ Acknowledgment  
Dataset inspired by real-world agricultural data.
