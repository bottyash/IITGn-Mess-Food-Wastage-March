import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ─── Style Constants ─────────────────────────────────────────────────────────
COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#FFC107", "#9C27B0"]
MEAL_COLORS = {"breakfast_waste": "#FF9800", "lunch_waste": "#2196F3",
               "snacks_waste": "#4CAF50", "dinner_waste": "#E91E63"}
BG_COLOR = "#FAFAFA"
GRID_ALPHA = 0.3


def _style_ax(ax, title, xlabel="", ylabel=""):
    """Apply consistent styling to an axis."""
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=GRID_ALPHA, linestyle="--")
    ax.set_facecolor(BG_COLOR)


class WasteEDA:

    def __init__(self, df):
        self.df = df.copy()
        self.meal_cols = [
            "breakfast_waste", "lunch_waste",
            "snacks_waste", "dinner_waste"
        ]
        # Ensure date is datetime
        if "date" in self.df.columns:
            self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
            self.df["day_name"] = self.df["date"].dt.day_name()
            self.df["is_weekend"] = self.df["date"].dt.dayofweek >= 5

    # ── 1. Overall Waste Trend ────────────────────────────────────────────
    def plot_trend(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        sorted_df = self.df.sort_values("date")
        ax.plot(sorted_df["date"], sorted_df["total_waste"],
                marker="o", color=COLORS[0], linewidth=2, markersize=5)
        ax.fill_between(sorted_df["date"], sorted_df["total_waste"],
                        alpha=0.15, color=COLORS[0])
        _style_ax(ax, "📈 Total Waste Over Time", "Date", "Waste (kg)")
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig

    # ── 2. Meal-wise Average Bar ──────────────────────────────────────────
    def plot_meal_distribution(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        means = self.df[self.meal_cols].mean()
        labels = [c.replace("_waste", "").title() for c in self.meal_cols]
        bars = ax.bar(labels, means, color=list(MEAL_COLORS.values()),
                      edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", fontweight="bold", fontsize=11)
        _style_ax(ax, "🍽️ Average Waste per Meal", "", "Avg Waste (kg)")
        fig.tight_layout()
        return fig

    # ── 3. Waste Distribution Histogram ───────────────────────────────────
    def plot_distribution(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(self.df["total_waste"], bins=12, color=COLORS[4],
                edgecolor="white", linewidth=1.2, alpha=0.85)
        mean_val = self.df["total_waste"].mean()
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=2,
                   label=f"Mean = {mean_val:.1f}")
        ax.legend(fontsize=11)
        _style_ax(ax, "📊 Waste Distribution", "Total Waste (kg)", "Frequency")
        fig.tight_layout()
        return fig

    # ── 4. Correlation Heatmap ────────────────────────────────────────────
    def plot_correlation(self):
        fig, ax = plt.subplots(figsize=(7, 6))
        cols = self.meal_cols + ["total_waste"]
        corr = self.df[cols].corr()
        cax = ax.matshow(corr, cmap="RdYlBu_r")
        fig.colorbar(cax, ax=ax, shrink=0.8)
        labels = [c.replace("_waste", "").title() for c in cols]
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="left", fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        # Annotate cells
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")
        ax.set_title("🔗 Correlation Heatmap", fontsize=14, fontweight="bold", pad=20)
        fig.tight_layout()
        return fig

    # ── 5. Top Waste Days ─────────────────────────────────────────────────
    def top_waste_days(self, n=5):
        top = self.df.sort_values("total_waste", ascending=False).head(n)
        result = top[["date", "total_waste"] + self.meal_cols].copy()
        result.columns = ["Date", "Total Waste", "Breakfast", "Lunch", "Snacks", "Dinner"]
        return result

    # ════════════════════════════════════════════════════════════════════════
    # NEW ANALYSES
    # ════════════════════════════════════════════════════════════════════════

    # ── 6. Day-of-Week Waste Pattern ──────────────────────────────────────
    def plot_day_of_week(self):
        fig, ax = plt.subplots(figsize=(9, 5))
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"]
        day_avg = self.df.groupby("day_name")["total_waste"].mean()
        day_avg = day_avg.reindex(day_order).dropna()
        colors = [COLORS[4] if d in ("Saturday", "Sunday") else COLORS[0]
                  for d in day_avg.index]
        bars = ax.bar(day_avg.index, day_avg.values, color=colors,
                      edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, day_avg.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", fontweight="bold", fontsize=10)
        _style_ax(ax, "📅 Waste by Day of Week", "", "Avg Waste (kg)")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        return fig

    # ── 7. Stacked Meal Area Chart ────────────────────────────────────────
    def plot_stacked_meals(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        sorted_df = self.df.sort_values("date")
        labels = [c.replace("_waste", "").title() for c in self.meal_cols]
        ax.stackplot(sorted_df["date"],
                     *[sorted_df[c] for c in self.meal_cols],
                     labels=labels, colors=list(MEAL_COLORS.values()),
                     alpha=0.8)
        ax.legend(loc="upper left", fontsize=10)
        _style_ax(ax, "📊 Meal-wise Waste Stacked Over Time", "Date", "Waste (kg)")
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig

    # ── 8. Rolling 7-Day Moving Average ──────────────────────────────────
    def plot_rolling_avg(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        sorted_df = self.df.sort_values("date").copy()
        sorted_df["rolling_7"] = sorted_df["total_waste"].rolling(window=7, min_periods=1).mean()
        ax.plot(sorted_df["date"], sorted_df["total_waste"],
                alpha=0.4, color=COLORS[0], label="Daily", linewidth=1)
        ax.plot(sorted_df["date"], sorted_df["rolling_7"],
                color=COLORS[1], linewidth=2.5, label="7-Day Moving Avg")
        ax.legend(fontsize=11)
        _style_ax(ax, "📉 Rolling 7-Day Average Waste", "Date", "Waste (kg)")
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig

    # ── 9. Meal Proportion Pie Chart ──────────────────────────────────────
    def plot_meal_pie(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        totals = self.df[self.meal_cols].sum()
        labels = [c.replace("_waste", "").title() for c in self.meal_cols]
        explode = [0.05] * len(labels)
        wedges, texts, autotexts = ax.pie(
            totals, labels=labels, autopct="%1.1f%%",
            colors=list(MEAL_COLORS.values()), explode=explode,
            shadow=True, startangle=140, textprops={"fontsize": 11}
        )
        for t in autotexts:
            t.set_fontweight("bold")
        ax.set_title("🥧 Meal-wise Waste Proportion", fontsize=14,
                      fontweight="bold", pad=15)
        fig.tight_layout()
        return fig

    # ── 10. Meal Box Plot ─────────────────────────────────────────────────
    def plot_meal_box(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        data = [self.df[c].dropna() for c in self.meal_cols]
        labels = [c.replace("_waste", "").title() for c in self.meal_cols]
        bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=True)
        for patch, color in zip(bp["boxes"], MEAL_COLORS.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        _style_ax(ax, "📦 Meal Waste Spread (Box Plot)", "", "Waste (kg)")
        fig.tight_layout()
        return fig

    # ── 11. Cumulative Waste ──────────────────────────────────────────────
    def plot_cumulative(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        sorted_df = self.df.sort_values("date")
        cumulative = sorted_df["total_waste"].cumsum()
        ax.plot(sorted_df["date"], cumulative,
                color=COLORS[2], linewidth=2.5, marker="o", markersize=4)
        ax.fill_between(sorted_df["date"], cumulative, alpha=0.15, color=COLORS[2])
        _style_ax(ax, "📈 Cumulative Waste Over Time", "Date", "Cumulative Waste (kg)")
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig

    # ── 12. Summary Statistics ────────────────────────────────────────────
    def get_summary_stats(self):
        cols = self.meal_cols + ["total_waste"]
        stats = self.df[cols].describe().T
        stats.columns = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
        stats.index = [c.replace("_waste", "").title() for c in cols]
        stats = stats.round(2)
        return stats.reset_index().rename(columns={"index": "Meal"})

    # ── 13. Weekday vs Weekend Comparison ─────────────────────────────────
    def plot_weekday_weekend(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        grouped = self.df.groupby("is_weekend")[self.meal_cols + ["total_waste"]].mean()
        labels = [c.replace("_waste", "").title() for c in self.meal_cols] + ["Total"]
        x = np.arange(len(labels))
        width = 0.35
        weekday_vals = grouped.loc[False].values if False in grouped.index else np.zeros(len(labels))
        weekend_vals = grouped.loc[True].values if True in grouped.index else np.zeros(len(labels))
        bars1 = ax.bar(x - width / 2, weekday_vals, width, label="Weekday",
                       color=COLORS[0], edgecolor="white")
        bars2 = ax.bar(x + width / 2, weekend_vals, width, label="Weekend",
                       color=COLORS[4], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.legend(fontsize=11)
        _style_ax(ax, "📊 Weekday vs Weekend Waste", "", "Avg Waste (kg)")
        fig.tight_layout()
        return fig