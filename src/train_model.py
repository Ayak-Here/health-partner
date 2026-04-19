import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

df_path = os.path.join(DATA_DIR, "early_disease_dataset.csv")
df = pd.read_csv(df_path)

encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender']) 

X = df[['Age', 'Gender', 'BloodPressure', 'Glucose', 'BMI']]
y = df['Risk']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nClassification report:")
print(classification_report(y_test, y_pred))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, os.path.join(MODEL_DIR, "diabetes_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(encoder, os.path.join(MODEL_DIR, "encoder.pkl"))

print("\nSaved files in 'models/' folder:")
print("diabetes_model.pkl")
print("scaler.pkl")
print("encoder.pkl")
