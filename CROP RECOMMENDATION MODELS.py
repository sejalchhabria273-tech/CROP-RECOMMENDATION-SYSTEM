import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

# ==========================================
# 1. SET WORKING DIRECTORY
# ==========================================
os.chdir("E:\BS CS IV files\AI\AI_Labs\project")

# ==========================================
# 2. LOAD DATA (CSV FILE)
# ==========================================
df = pd.read_csv("Crop_recommendation.csv")   # ✅ FIXED

print("Dataset Loaded Successfully ✅")
print(df.head())

# ==========================================
# 3. FEATURES & LABEL
# ==========================================
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

le = LabelEncoder()
y_class = le.fit_transform(df['label'])

# Regression target (simple dummy example)
y_reg = df['rainfall'] * 0.02

# ==========================================
# 4. TRAIN-TEST SPLIT
# ==========================================
X_train, X_test, y_train_c, y_test_c, y_train_r, y_test_r = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

# ==========================================
# 5. SCALING (for KNN)
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 6. MODELS
# ==========================================

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train_c)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train_c)
y_pred_knn = knn.predict(X_test_scaled)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train_r)
y_pred_reg = lr.predict(X_test)

# ==========================================
# 7. VISUALIZATION
# ==========================================
plt.figure(figsize=(18,5))

# Feature Importance
plt.subplot(1,3,1)
sns.barplot(x=X.columns, y=rf.feature_importances_)
plt.title("Feature Importance")

# Confusion Matrix
plt.subplot(1,3,2)
cm = confusion_matrix(y_test_c, y_pred_knn)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")

# Regression Plot
plt.subplot(1,3,3)
plt.scatter(y_test_r, y_pred_reg)
plt.plot([y_test_r.min(), y_test_r.max()],
         [y_test_r.min(), y_test_r.max()], 'r--')
plt.title("Actual vs Predicted Yield")

plt.tight_layout()
plt.show()

# ==========================================
# 8. RESULTS
# ==========================================
print("\n--- RESULTS ---")
print("kNN Accuracy:", round(accuracy_score(y_test_c, y_pred_knn)*100, 2), "%")
print("Regression MSE:", round(mean_squared_error(y_test_r, y_pred_reg), 4))