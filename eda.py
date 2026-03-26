import matplotlib.pyplot as plt
import pandas as pd


class WasteEDA:

    def __init__(self, df):
        self.df = df.copy()

    # 1. Overall Waste Trend
    def plot_trend(self):
        fig, ax = plt.subplots()

        self.df.sort_values("date").plot(
            x="date",
            y="total_waste",
            ax=ax
        )

        ax.set_title("Total Waste Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Waste")

        return fig

    # 2. Meal-wise Distribution
    def plot_meal_distribution(self):
        fig, ax = plt.subplots()

        meal_cols = [
            "breakfast_waste",
            "lunch_waste",
            "snacks_waste",
            "dinner_waste"
        ]

        self.df[meal_cols].mean().plot(kind="bar", ax=ax)

        ax.set_title("Average Waste per Meal")
        ax.set_ylabel("Waste")

        return fig

    # 3. Waste Distribution Histogram
    def plot_distribution(self):
        fig, ax = plt.subplots()

        self.df["total_waste"].plot(kind="hist", bins=15, ax=ax)

        ax.set_title("Waste Distribution")
        ax.set_xlabel("Waste")

        return fig

    # 4. Correlation Heatmap (Manual since no seaborn)
    def plot_correlation(self):
        fig, ax = plt.subplots()

        corr = self.df[[
            "breakfast_waste",
            "lunch_waste",
            "snacks_waste",
            "dinner_waste",
            "total_waste"
        ]].corr()

        cax = ax.matshow(corr)

        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45)
        ax.set_yticklabels(corr.columns)

        fig.colorbar(cax)

        ax.set_title("Correlation Heatmap")

        return fig

    # 5. Top Waste Days
    def top_waste_days(self, n=5):
        return self.df.sort_values("total_waste", ascending=False)[
            ["date", "total_waste"]
        ].head(n)