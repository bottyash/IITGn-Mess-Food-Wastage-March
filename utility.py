import pandas as pd
import matplotlib.pyplot as plt
from model import load_and_merge


def waste_trend_plot():
    df = pd.read_csv("food-wastage.csv")

    plt.figure()
    plt.plot(df["Date"], df["Total_waste"])
    plt.xticks(rotation=45)
    plt.title("Total Waste Trend")
    plt.tight_layout()

    path = "trend.png"
    plt.savefig(path)
    plt.close()

    return path


def meal_wise_plot():
    df = pd.read_csv("food-wastage.csv")

    plt.figure()
    plt.plot(df["Breakfast_waste"], label="Breakfast")
    plt.plot(df["Lunch_waste"], label="Lunch")
    plt.plot(df["Snacks_waste"], label="Snacks")
    plt.plot(df["Dinner_waste"], label="Dinner")
    plt.legend()
    plt.title("Meal-wise Waste")

    path = "meals.png"
    plt.savefig(path)
    plt.close()

    return path


def menu_impact():
    df = load_and_merge()

    impact = df.groupby("Lunch_item")["Total_waste"].mean().sort_values(ascending=False)

    return impact.head(5).to_string()