import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv("data/student_data.csv")

print("\nFIRST 5 ROWS:")
print(data.head())

# ==========================================
# FEATURES (INPUTS)
# ==========================================

X = data[['studytime', 'failures', 'absences', 'G1', 'G2']]

# ==========================================
# TARGET (OUTPUT)
# ==========================================

y = data['G3']

# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# CREATE MODEL
# ==========================================

model = LinearRegression()

# ==========================================
# TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)

# ==========================================
# MAKE PREDICTIONS
# ==========================================

predictions = model.predict(X_test)

print("\nPREDICTIONS:")
print(predictions[:10])

# ==========================================
# MODEL EVALUATION
# ==========================================

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("-------------------")
print(f"R2 Score: {r2:.4f}")
print(f"Accuracy: {r2 * 100:.2f}%")
print(f"Mean Absolute Error: {mae:.2f}")

# ==========================================
# USER INPUT
# ==========================================

print("\nEnter Student Details")

studytime = float(input("Study Time (1-4): "))
failures = float(input("Past Failures: "))
absences = float(input("Absences: "))
g1 = float(input("G1 Grade: "))
g2 = float(input("G2 Grade: "))

# ==========================================
# CREATE INPUT DATAFRAME
# ==========================================

custom_data = pd.DataFrame(
    [[studytime, failures, absences, g1, g2]],
    columns=[
        'studytime',
        'failures',
        'absences',
        'G1',
        'G2'
    ]
)

# ==========================================
# PREDICT FINAL GRADE
# ==========================================

predicted_score = model.predict(custom_data)

print("\n========================")
print(f"Predicted Final Grade (G3): {predicted_score[0]:.2f}")
print("========================")

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(model, "models/model.pkl")

print("\nModel saved successfully!")
print("Saved at: models/model.pkl")

# ==========================================
# VISUALIZATION
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    data['G2'],
    data['G3']
)

plt.xlabel("G2 Grade")
plt.ylabel("Final Grade (G3)")
plt.title("G2 Grade vs Final Grade")

plt.grid(True)

plt.show()
