import pandas as pd, numpy as np, pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def train(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)

    models = {
        "LR": LogisticRegression(max_iter=500),
        "RF": RandomForestClassifier(),
        "XGB": XGBClassifier(eval_metric='logloss')
    }

    best = None
    best_acc = 0
    for name, model in models.items():
        model.fit(Xtr, ytr)
        acc = accuracy_score(yte, model.predict(Xte))
        if acc > best_acc:
            best_acc = acc
            best = model

    return best, sc



# ----------DIABETES DATASET-----------------

diabetes = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
dm, ds = train(diabetes, "Outcome")


# ----------HEART DISEASE DATASET-------------

heart = pd.read_csv("heart_disease_uci.csv")

# Encode ALL categorical columns
categorical_cols = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
for col in categorical_cols:
    if col in heart.columns:
        heart[col] = heart[col].astype('category').cat.codes

# Drop ID column
if "id" in heart.columns:
    heart = heart.drop(columns=["id"])

# Drop rows with missing values (critical)
heart = heart.dropna()

# Set target
heart_target = "num"
hm, hs = train(heart, heart_target)





#---------- KIDNEY DISEASE DATASET-----------

kidney = pd.read_csv("kidney_disease.csv")

# Replace "?" or empty strings with NaN
kidney.replace("?", np.nan, inplace=True)
kidney.replace("", np.nan, inplace=True)

# Convert classification text cleanly
kidney["classification"] = kidney["classification"].astype(str).str.lower().str.strip()

# Map labels safely
kidney["classification"] = kidney["classification"].map({
    "ckd": 1,
    "notckd": 0,
    "ckd\t": 1,     # common bug in dataset
    "notckd\t": 0,  # extra tab
})
# REMOVE ID :
if "id" in kidney.columns:
    kidney = kidney.drop(columns=["id"])

# Drop ANY remaining rows with missing target
kidney = kidney.dropna(subset=["classification"])

# Convert all categorical columns to numeric
for col in kidney.columns:
    kidney[col] = kidney[col].astype("category").cat.codes

km, ks = train(kidney, "classification")



# ---------SAVE ALL TRAINED MODELS-------------

pickle.dump((dm, ds), open("models/diabetes_model.pkl", "wb"))
pickle.dump((hm, hs), open("models/heart_model.pkl", "wb"))
pickle.dump((km, ks), open("models/kidney_model.pkl", "wb"))

print("Models saved!")
