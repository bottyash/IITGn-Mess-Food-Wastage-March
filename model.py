import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


class WasteModel:
    def __init__(self):
        self.pipeline = Pipeline([
            ("vectorizer", CountVectorizer()),
            ("model", RandomForestRegressor(n_estimators=150, random_state=42))
        ])
        self.df = None
        self.trained = False

    def train(self, df):
        df = df.dropna(subset=["menu_text", "total_waste"])

        self.df = df
        X = df["menu_text"]
        y = df["total_waste"]

        self.pipeline.fit(X, y)
        self.trained = True

    def predict(self, breakfast, lunch, snacks, dinner):
        if not self.trained:
            return 0

        text = f"{breakfast} {lunch} {snacks} {dinner}"
        pred = self.pipeline.predict([text])[0]
        return round(float(pred), 2)

    def get_trend_plot(self):
        fig, ax = plt.subplots()

        self.df.sort_values("date").plot(
            x="date",
            y="total_waste",
            ax=ax
        )

        ax.set_title("Waste Trend")
        ax.set_ylabel("Waste")
        return fig

    def get_meal_plot(self):
        fig, ax = plt.subplots()

        self.df[[
            "breakfast_waste",
            "lunch_waste",
            "snacks_waste",
            "dinner_waste"
        ]].mean().plot(kind="bar", ax=ax)

        ax.set_title("Average Meal-wise Waste")
        return fig

    def get_food_impact(self):
        vectorizer = self.pipeline.named_steps["vectorizer"]
        model = self.pipeline.named_steps["model"]

        features = vectorizer.get_feature_names_out()
        importance = model.feature_importances_

        df_imp = pd.DataFrame({
            "food": features,
            "impact": importance
        }).sort_values("impact", ascending=False).head(15)

        fig, ax = plt.subplots()
        df_imp.plot(x="food", y="impact", kind="bar", ax=ax)

        ax.set_title("Top Waste-Contributing Items")
        return fig

    def get_insight(self, value):
        if value < 20:
            return "Low Waste"
        elif value < 50:
            return "Moderate Waste"
        else:
            return "High Waste"