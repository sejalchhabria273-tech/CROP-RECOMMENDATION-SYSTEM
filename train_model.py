import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("crop_recommendation.csv")

features = ['N','P','K','temperature','humidity','ph','rainfall']
X = df[features]
y = df['label']

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Save model and label encoder
joblib.dump(model, "models/crop_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")

print("✅ Model saved successfully in 'models/' folder")
