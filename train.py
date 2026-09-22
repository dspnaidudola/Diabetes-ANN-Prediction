import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# 1. LOAD DATASET

df = pd.read_csv("diabetes.csv")

print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())


# 2. DATA CLEANING

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

df = df.drop_duplicates()



# 3. HANDLE INVALID ZERO VALUES


zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

df[zero_columns] = df[zero_columns].replace(0, np.nan)

print("\nMissing Values After Zero Check:")
print(df.isnull().sum())



# 4. FEATURES AND TARGET


X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# 5. TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)



# 6. IMPUTATION


imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

joblib.dump(imputer, "imputer.pkl")



# 7. FEATURE SCALING


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")



# 8. ANN MODEL


model = Sequential()

model.add(
    Dense(
        32,
        activation="relu",
        input_shape=(8,)
    )
)

model.add(Dropout(0.30))

model.add(
    Dense(
        16,
        activation="relu"
    )
)

model.add(Dropout(0.20))

model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)



# 9. COMPILE MODEL

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# 10. EARLY STOPPING

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)


# 11. TRAIN MODEL

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=150,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)


# 12. EVALUATE MODEL

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nTest Accuracy:", accuracy)


# 13. SAVE MODEL

model.save("diabetes_ann_model.keras")

print("\nModel saved successfully.")
print("Created:")
print("diabetes_ann_model.keras")
print("imputer.pkl")
print("scaler.pkl")