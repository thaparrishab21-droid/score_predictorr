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

print("\nDATASET:")
print(data)

# ==========================================
# FEATURES (INPUTS)
# ==========================================

X = data[['StudyHours', 'SleepHours', 'Attendance']]

# ==========================================
# TARGET (OUTPUT)
# ==========================================

y = data['Score']

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
print(predictions)

# ==========================================
# MODEL EVALUATION
# ==========================================

r2 = r2_score(y_test, predictions)

mae = mean_absolute_error(y_test, predictions)

print("\nR2 Score:", r2)
print("Mean Absolute Error:", mae)

# ==========================================
# USER INPUT
# ==========================================

study = float(input("\nEnter study hours: "))
sleep = float(input("Enter sleep hours: "))
attendance = float(input("Enter attendance percentage: "))

# Convert input into DataFrame
custom_data = pd.DataFrame(
    [[study, sleep, attendance]],
    columns=['StudyHours', 'SleepHours', 'Attendance']
)

# ==========================================
# PREDICT USER SCORE
# ==========================================

predicted_score = model.predict(custom_data)

print(f"\nPredicted Score: {predicted_score[0]:.2f}")

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(model, "models/model.pkl")

print("\nModel saved successfully!")

# ==========================================
# VISUALIZATION
# ==========================================

# Scatter plot (actual data)
plt.scatter(
    data['StudyHours'],
    data['Score']
)

# Regression line
plt.plot(
    data['StudyHours'],
    model.predict(X)
)

# Labels and title
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Study Hours vs Score")

# Show graph
plt.show()