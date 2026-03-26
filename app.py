import gradio as gr
from utility import *
from model import WasteModel
from eda import WasteEDA

# ─── LOAD & PROCESS DATA ─────────────────────────────────────────────────────
menu_df = load_csv("clean_menu.csv")
waste_df = load_csv("food-wastage.csv")

menu_df = process_menu(menu_df)
menu_df = map_day_to_date(menu_df)

df = merge_data(menu_df, waste_df)
df = create_text_feature(df)

# ─── INITIALIZE EDA & MODEL ──────────────────────────────────────────────────
eda = WasteEDA(df)
model = WasteModel()
model.train(df)


# ─── PREDICTION FUNCTION ─────────────────────────────────────────────────────
def predict_waste(b, l, s, d):
    pred = model.predict(b, l, s, d)
    insight = model.get_insight(pred)
    return pred, insight


# ═══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════════
with gr.Blocks(
    title="IIT Gandhinagar Mess Waste Insights",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="orange"),
) as app:

    gr.Markdown(
        """
        # IIT Gandhinagar Mess Waste Insights for Jaiswal Mess at IIT Gandhinagar for the month of March 2026
        > Predict, analyze, and reduce food waste in institutional mess facilities.
        """
    )

    # ── Tab 1: Prediction ────────────────────────────────────────────────
    with gr.Tab("Prediction"):
        gr.Markdown("### Enter today's menu to predict total food waste")
        with gr.Row():
            b = gr.Textbox(label="Breakfast", placeholder="e.g. poha, tea, bread")
            l = gr.Textbox(label="Lunch", placeholder="e.g. dal, rice, chapati")
        with gr.Row():
            s = gr.Textbox(label="Snacks", placeholder="e.g. samosa, chai")
            d = gr.Textbox(label="Dinner", placeholder="e.g. paneer, rice, roti")

        btn = gr.Button("Predict Waste", variant="primary")

        with gr.Row():
            out1 = gr.Number(label="Predicted Waste (kg)")
            out2 = gr.Textbox(label="Insight")

        btn.click(predict_waste, inputs=[b, l, s, d], outputs=[out1, out2])

        gr.Markdown(f"*Best model: **{model.best_name}***")

    # ── Tab 2: Model Performance ─────────────────────────────────────────
    with gr.Tab("Model Performance"):
        gr.Markdown("### Model Evaluation & Comparison")

        gr.Markdown("#### Metrics Table (best model)")
        gr.Dataframe(value=model.get_metrics_df(), label="Model Metrics")

        gr.Markdown("---")
        gr.Markdown("#### Model Visualization")

        with gr.Row():
            gr.Plot(model.plot_model_comparison)
            gr.Plot(model.plot_cv_scores)

        with gr.Row():
            gr.Plot(model.plot_actual_vs_predicted)
            gr.Plot(model.plot_residuals)

        gr.Plot(model.plot_prediction_error)

    # ── Tab 3: Trends ────────────────────────────────────────────────────
    with gr.Tab("Trends"):
        gr.Markdown("### Waste Trend Analysis")
        gr.Plot(model.get_trend_plot)

    # ── Tab 4: Meal Analysis ─────────────────────────────────────────────
    with gr.Tab("Meal Analysis"):
        gr.Markdown("### Meal-wise Waste Breakdown")
        gr.Plot(model.get_meal_plot)

    # ── Tab 5: Menu Impact ───────────────────────────────────────────────
    with gr.Tab("Menu Impact"):
        gr.Markdown("### Food Items Contributing Most to Waste")
        gr.Plot(model.get_food_impact)

    # ── Tab 6: EDA Insights ──────────────────────────────────────────────
    with gr.Tab("EDA Insights"):
        gr.Markdown(
            """
            ## Exploratory Data Analysis
            Comprehensive analysis of mess food waste patterns.
            """
        )

        gr.Markdown("### Summary Statistics")
        gr.Dataframe(value=eda.get_summary_stats(), label="Descriptive Statistics")

        gr.Markdown("---")
        gr.Markdown("### Trend Analysis")
        with gr.Row():
            gr.Plot(eda.plot_trend)
            gr.Plot(eda.plot_rolling_avg)

        gr.Markdown("---")
        gr.Markdown("### Meal-wise Analysis")
        with gr.Row():
            gr.Plot(eda.plot_meal_distribution)
            gr.Plot(eda.plot_meal_pie)

        with gr.Row():
            gr.Plot(eda.plot_meal_box)
            gr.Plot(eda.plot_stacked_meals)

        gr.Markdown("---")
        gr.Markdown("### Distribution & Patterns")
        with gr.Row():
            gr.Plot(eda.plot_distribution)
            gr.Plot(eda.plot_correlation)

        with gr.Row():
            gr.Plot(eda.plot_day_of_week)
            gr.Plot(eda.plot_weekday_weekend)

        gr.Markdown("---")
        gr.Markdown("### Cumulative Analysis")
        gr.Plot(eda.plot_cumulative)

        gr.Markdown("---")
        gr.Markdown("### Top Waste Days")
        top_days = gr.Dataframe(label="Top 5 Highest Waste Days")
        btn2 = gr.Button("Show Top Waste Days", variant="secondary")
        btn2.click(lambda: eda.top_waste_days(), outputs=top_days)


app.launch(server_name="localhost", server_port=7860)