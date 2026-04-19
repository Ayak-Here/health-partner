import numpy as np
import pandas as pd

np.random.seed(42)
size = 1000

age = np.random.randint(20, 80, size)
gender = np.random.choice(["Male", "Female"], size)
bp = np.random.randint(90, 180, size)
glucose = np.random.randint(70, 200, size)
bmi = np.random.uniform(15, 40, size)

risk = []
for bmi_val, bp_val, gl in zip(bmi, bp, glucose):
    if gl > 140 or bmi_val > 30 or bp_val > 140:
        risk.append("High Risk")
    else:
        risk.append("Low Risk")

df = pd.DataFrame({
    "Age": age,
    "Gender": gender,
    "BloodPressure": bp,
    "Glucose": glucose,
    "BMI": bmi,
    "Risk": risk
})

df.to_csv("early_disease_dataset.csv", index=False)
print("Saved early_disease_dataset.csv with", len(df), "rows")
