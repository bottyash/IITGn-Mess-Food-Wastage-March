import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import pickle

# -------- MENU PARSER --------
def parse_menu():
    df = pd.read_excel("menu.xlsx", header=None)

    structured_data = []
    i = 0

    while i < len(df):
        row = df.iloc[i]

        if isinstance(row[0], str) and row[0].strip() not in ["Breakfast", None]:
            day = row[0]

            breakfast = df.iloc[i+2:i+6, 0].dropna().astype(str).tolist()
            lunch = df.iloc[i+2:i+6, 1].dropna().astype(str).tolist()
            snacks = df.iloc[i+2:i+6, 2].dropna().astype(str).tolist()
            dinner = df.iloc[i+2:i+6, 3].dropna().astype(str).tolist()

            structured_data.append({
                "Day": day,
                "Breakfast_item": " | ".join(breakfast),
                "Lunch_item": " | ".join(lunch),
                "Snacks_item": " | ".join(snacks),
                "Dinner_item": " | ".join(dinner),
            })

            i += 6
        else:
            i += 1

    return pd.DataFrame(structured_data)


# -------- DATE MAPPING --------
def add_dates(menu_df):
    start_date = pd.to_datetime("2026-03-01")

    menu_df["Date"] = [
        start_date + pd.Timedelta(days=i)
        for i in range(len(menu_df))
    ]

    return menu_df


# -------- MERGE --------
def load_and_merge():
    waste_df = pd.read_csv("food-wastage.csv")
    waste_df["Date"] = pd.to_datetime(waste_df["Date"])

    menu_df = parse_menu()
    menu_df = add_dates(menu_df)

    df = pd.merge(waste_df, menu_df, on="Date")

    return df


# -------- TRAIN --------
def train_model():
    df = load_and_merge()

    X = df[["Breakfast_item","Lunch_item","Snacks_item","Dinner_item"]]
    y = df["Total_waste"]

    encoder = OneHotEncoder(handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_encoded, y)

    with open("model.pkl", "wb") as f:
        pickle.dump((model, encoder), f)


# -------- PREDICT --------
def predict(menu):
    with open("model.pkl", "rb") as f:
        model, encoder = pickle.load(f)

    df = pd.DataFrame([menu])
    X = encoder.transform(df)

    return model.predict(X)[0]