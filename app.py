import gradio as gr
import pandas as pd
from model import train_model, predict
from utils import waste_trend_plot, meal_wise_plot, menu_impact

# Train model once
train_model()

# Load menu for dropdowns
menu_df = pd.read_excel("menu.xlsx", header=None)

# Extract unique food items (basic fallback)
options = ["idli", "poha", "paratha", "rice", "dal", "paneer", "khichdi", "samosa", "tea", "pakoda"]

def predict_waste(b, l, s, d):
    menu = {
        "Breakfast_item": b,
        "Lunch_item": l,
        "Snacks_item": s,
        "Dinner_item": d
    }

    pred = predict(menu)

    if pred > 35:
        insight = " High waste expected"
    elif pred < 25:
        insight = "Low waste (good menu)"
    else:
        insight = "Moderate waste"

    return f"{round(pred,2)} kg", insight


with gr.Blocks(title="Smart Mess AI") as app:

    gr.Markdown("#Smart Mess Optimization System")

    with gr.Tab("Prediction"):
        b = gr.Dropdown(options, label="Breakfast")
        l = gr.Dropdown(options, label="Lunch")
        s = gr.Dropdown(options, label="Snacks")
        d = gr.Dropdown(options, label="Dinner")

        btn = gr.Button("Predict")

        out = gr.Textbox(label="Predicted Waste")
        insight = gr.Textbox(label="Insight")

        btn.click(predict_waste, inputs=[b,l,s,d], outputs=[out,insight])

    with gr.Tab("Trends"):
        gr.Image(waste_trend_plot)

    with gr.Tab("Meal Analysis"):
        gr.Image(meal_wise_plot)

    with gr.Tab("Menu Impact"):
        gr.Textbox(menu_impact())

app.launch()