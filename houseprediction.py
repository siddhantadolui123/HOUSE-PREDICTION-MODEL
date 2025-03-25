import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("C:\\project\\CODEXINTERN PYTHON TASK\\Housing.csv")

# displaying dataset info
print(df.head())
print(df.info())
print(df.describe())
df = df.dropna() #handles any missing value from dataset

categorical_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", 
                    "airconditioning", "prefarea", "furnishingstatus"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# define features and target variable
X = df.drop(columns=["price"])
y = df["price"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# train the model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# examine the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")  #measure difference between actual and predicted prices lower thr mae is better
print(f"Mean Squared Error (MSE): {mse}")   #same as mse lower the mse is better
print(f"R-squared Score (R²): {r2}")         # explain variation in house prices

# Function to predict house price
def predict_house_price(features):
    features_df = pd.DataFrame([features])
    for col in categorical_cols:
        if col in features_df:
            features_df[col] = label_encoders[col].transform(features_df[col])
    features_scaled = scaler.transform(features_df)
    return model.predict(features_scaled)[0]

# update accordinf to the requirements of the user
new_house = {
    "area": 10000,
    "bedrooms": 2,
    "bathrooms": 2,
    "stories": 1,
    "mainroad": "yes",
    "guestroom": "yes",
    "basement": "no",
    "hotwaterheating": "no",
    "airconditioning": "yes",
    "parking": 1,
    "prefarea": "yes",
    "furnishingstatus": "furnished"
}
predicted_price = predict_house_price(new_house)
print(f"Predicted House Price: {predicted_price}")
