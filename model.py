import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─── Style Constants ─────────────────────────────────────────────────────────
COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#FFC107", "#9C27B0"]
BG_COLOR = "#FAFAFA"
GRID_ALPHA = 0.3


def _style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=GRID_ALPHA, linestyle="--")
    ax.set_facecolor(BG_COLOR)


class WasteModel:
    def __init__(self):
        self.models = {
            "Random Forest": Pipeline([
                ("vectorizer", TfidfVectorizer(max_features=200)),
                ("model", RandomForestRegressor(n_estimators=200, random_state=42))
            ]),
            "Gradient Boosting": Pipeline([
                ("vectorizer", TfidfVectorizer(max_features=200)),
                ("model", GradientBoostingRegressor(n_estimators=150, random_state=42,
                                                     learning_rate=0.1, max_depth=4))
            ]),
            "Ridge Regression": Pipeline([
                ("vectorizer", TfidfVectorizer(max_features=200)),
                ("model", Ridge(alpha=1.0))
            ]),
        }
        self.best_name = None
        self.best_pipeline = None
        self.df = None
        self.trained = False
        self.metrics = {}        # {name: {r2, mae, rmse}}
        self.cv_scores = {}      # {name: array of cv scores}
        self.y_true = None
        self.y_pred = None

    def train(self, df):
        df = df.dropna(subset=["menu_text", "total_waste"])
        self.df = df.copy()
        X = df["menu_text"]
        y = df["total_waste"]

        best_r2 = -np.inf

        for name, pipeline in self.models.items():
            # Cross-validation (need at least 2 folds)
            n_samples = len(X)
            if n_samples < 5:
                cv_strategy = LeaveOneOut()
            else:
                cv_strategy = min(5, n_samples)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(pipeline, X, y, cv=cv_strategy,
                                         scoring="r2")
            self.cv_scores[name] = scores

            # Fit on full data
            pipeline.fit(X, y)
            preds = pipeline.predict(X)

            r2 = r2_score(y, preds)
            mae = mean_absolute_error(y, preds)
            rmse = np.sqrt(mean_squared_error(y, preds))

            self.metrics[name] = {
                "R²": round(r2, 4),
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "CV R² (mean)": round(scores.mean(), 4),
                "CV R² (std)": round(scores.std(), 4),
            }

            if scores.mean() > best_r2:
                best_r2 = scores.mean()
                self.best_name = name
                self.best_pipeline = pipeline

        # Store predictions from best model
        self.y_true = y.values
        self.y_pred = self.best_pipeline.predict(X)
        self.trained = True

    def predict(self, breakfast, lunch, snacks, dinner):
        if not self.trained:
            return 0
        text = f"{breakfast} {lunch} {snacks} {dinner}"
        pred = self.best_pipeline.predict([text])[0]
        return round(float(pred), 2)

    def get_insight(self, value):
        if self.df is not None and len(self.df) > 0:
            q25 = self.df["total_waste"].quantile(0.25)
            q75 = self.df["total_waste"].quantile(0.75)
        else:
            q25, q75 = 20, 40

        if value < q25:
            return f"✅ Low Waste (below 25th percentile: {q25:.1f} kg)"
        elif value < q75:
            return f"⚠️ Moderate Waste (between {q25:.1f}–{q75:.1f} kg)"
        else:
            return f"🚨 High Waste (above 75th percentile: {q75:.1f} kg)"

    def get_metrics_df(self):
        """Return a DataFrame of all model metrics."""
        rows = []
        for name, m in self.metrics.items():
            best_flag = " ⭐" if name == self.best_name else ""
            rows.append({
                "Model": name + best_flag,
                "R²": m["R²"],
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "CV R² (mean)": m["CV R² (mean)"],
                "CV R² (std)": m["CV R² (std)"],
            })
        return pd.DataFrame(rows)

    # ══════════════════════════════════════════════════════════════════════
    # EXISTING PLOTS (improved styling)
    # ══════════════════════════════════════════════════════════════════════

    def get_trend_plot(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        sorted_df = self.df.sort_values("date")
        ax.plot(sorted_df["date"], sorted_df["total_waste"],
                marker="o", color=COLORS[0], linewidth=2, markersize=5)
        ax.fill_between(sorted_df["date"], sorted_df["total_waste"],
                        alpha=0.15, color=COLORS[0])
        _style_ax(ax, "📈 Waste Trend Over Time", "Date", "Waste (kg)")
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig

    def get_meal_plot(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        meal_cols = ["breakfast_waste", "lunch_waste", "snacks_waste", "dinner_waste"]
        means = self.df[meal_cols].mean()
        labels = [c.replace("_waste", "").title() for c in meal_cols]
        colors = ["#FF9800", "#2196F3", "#4CAF50", "#E91E63"]
        bars = ax.bar(labels, means, color=colors, edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", fontweight="bold")
        _style_ax(ax, "🍽️ Average Meal-wise Waste", "", "Avg Waste (kg)")
        fig.tight_layout()
        return fig

    def get_food_impact(self):
        vectorizer = self.best_pipeline.named_steps["vectorizer"]
        model = self.best_pipeline.named_steps["model"]
        features = vectorizer.get_feature_names_out()

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_)
        else:
            importance = np.zeros(len(features))

        df_imp = pd.DataFrame({
            "food": features, "impact": importance
        }).sort_values("impact", ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(df_imp["food"][::-1], df_imp["impact"][::-1],
                color=COLORS[1], edgecolor="white")
        _style_ax(ax, "🍕 Top Waste-Contributing Food Items", "Impact Score", "")
        fig.tight_layout()
        return fig

    # ══════════════════════════════════════════════════════════════════════
    # NEW MODEL VISUALIZATION PLOTS
    # ══════════════════════════════════════════════════════════════════════

    def plot_actual_vs_predicted(self):
        """Scatter plot: Actual vs Predicted waste."""
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(self.y_true, self.y_pred, color=COLORS[0], alpha=0.7,
                   edgecolors="white", s=80, zorder=3)
        # Perfect prediction line
        lims = [min(self.y_true.min(), self.y_pred.min()) - 2,
                max(self.y_true.max(), self.y_pred.max()) + 2]
        ax.plot(lims, lims, "r--", linewidth=2, label="Perfect Prediction", zorder=2)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.legend(fontsize=11)
        _style_ax(ax, f"🎯 Actual vs Predicted ({self.best_name})",
                  "Actual Waste (kg)", "Predicted Waste (kg)")
        fig.tight_layout()
        return fig

    def plot_residuals(self):
        """Residual plot to check model error patterns."""
        fig, ax = plt.subplots(figsize=(10, 5))
        residuals = self.y_true - self.y_pred
        ax.scatter(self.y_pred, residuals, color=COLORS[4], alpha=0.7,
                   edgecolors="white", s=80, zorder=3)
        ax.axhline(y=0, color="red", linewidth=2, linestyle="--", zorder=2)
        mean_res = residuals.mean()
        std_res = residuals.std()
        ax.axhline(y=mean_res + 2 * std_res, color="orange", linewidth=1,
                   linestyle=":", label=f"±2σ ({2*std_res:.1f})")
        ax.axhline(y=mean_res - 2 * std_res, color="orange", linewidth=1,
                   linestyle=":")
        ax.legend(fontsize=11)
        _style_ax(ax, "📊 Residual Plot", "Predicted Waste (kg)", "Residual (kg)")
        fig.tight_layout()
        return fig

    def plot_model_comparison(self):
        """Bar chart comparing R² across all models."""
        fig, ax = plt.subplots(figsize=(9, 5))
        names = list(self.metrics.keys())
        r2_train = [self.metrics[n]["R²"] for n in names]
        r2_cv = [self.metrics[n]["CV R² (mean)"] for n in names]
        cv_std = [self.metrics[n]["CV R² (std)"] for n in names]

        x = np.arange(len(names))
        width = 0.35

        bars1 = ax.bar(x - width / 2, r2_train, width, label="Train R²",
                       color=COLORS[0], edgecolor="white")
        bars2 = ax.bar(x + width / 2, r2_cv, width, label="CV R² (mean)",
                       color=COLORS[2], edgecolor="white", yerr=cv_std, capsize=5)

        # Highlight best
        best_idx = names.index(self.best_name)
        bars2[best_idx].set_edgecolor("gold")
        bars2[best_idx].set_linewidth(3)

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=11)
        ax.legend(fontsize=11)
        _style_ax(ax, "🏆 Model Comparison", "", "R² Score")
        fig.tight_layout()
        return fig

    def plot_cv_scores(self):
        """Box plot of cross-validation scores per model."""
        fig, ax = plt.subplots(figsize=(8, 5))
        data = list(self.cv_scores.values())
        labels = list(self.cv_scores.keys())
        bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
        for patch, color in zip(bp["boxes"], COLORS[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        _style_ax(ax, "📦 Cross-Validation Score Distribution", "", "R² Score")
        ax.tick_params(axis="x", rotation=10)
        fig.tight_layout()
        return fig

    def plot_prediction_error(self):
        """Bar chart showing per-sample prediction errors."""
        fig, ax = plt.subplots(figsize=(10, 5))
        errors = self.y_true - self.y_pred
        colors_arr = [COLORS[2] if e >= 0 else COLORS[1] for e in errors]
        ax.bar(range(len(errors)), errors, color=colors_arr, edgecolor="white")
        ax.axhline(y=0, color="black", linewidth=0.8)
        _style_ax(ax, "📊 Per-Sample Prediction Error", "Sample Index", "Error (kg)")
        fig.tight_layout()
        return fig