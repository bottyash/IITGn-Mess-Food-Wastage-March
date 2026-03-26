import pandas as pd
import re
from datetime import datetime, timedelta


def load_csv(file_path):
    try:
        return pd.read_csv(file_path, encoding="utf-8")
    except Exception:
        return pd.read_csv(file_path, encoding="latin1")


def clean_columns(df):
    df.columns = [col.strip().lower() for col in df.columns]
    return df


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[+/,\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def process_menu(menu_df):
    menu_df = clean_columns(menu_df)

    grouped = (
        menu_df.groupby(["day", "meal_type"])["food_item"]
        .apply(lambda x: " ".join(x))
        .reset_index()
    )

    pivot = grouped.pivot(index="day", columns="meal_type", values="food_item").reset_index()

    pivot.columns.name = None
    pivot.columns = [col.lower() for col in pivot.columns]

    for col in ["breakfast", "lunch", "snacks", "dinner"]:
        if col not in pivot.columns:
            pivot[col] = ""

    for col in ["breakfast", "lunch", "snacks", "dinner"]:
        pivot[col] = pivot[col].apply(clean_text)

    return pivot


def map_day_to_date(menu_df, start_date="2026-03-01"):
    start = datetime.strptime(start_date, "%Y-%m-%d")

    day_mapping = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    menu_df["day"] = menu_df["day"].str.lower()

    menu_df["date"] = menu_df["day"].map(
        lambda d: start + timedelta(days=day_mapping.get(d, 0))
    )

    return menu_df


def get_day_name(date):
    """Extract the day name from a date."""
    if pd.isna(date):
        return ""
    return pd.to_datetime(date).strftime("%A")


def merge_data(menu_df, waste_df):
    waste_df = clean_columns(waste_df)
    waste_df["date"] = pd.to_datetime(waste_df["date"], format="%d-%m-%y", errors="coerce")

    merged = pd.merge(menu_df, waste_df, on="date", how="inner")
    return merged


def create_text_feature(df):
    df["menu_text"] = (
        df["breakfast"] + " " +
        df["lunch"] + " " +
        df["snacks"] + " " +
        df["dinner"]
    )
    return df