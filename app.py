import gradio as gr
from utility import *
from model import WasteModel

# LOAD FILES (your existing names)
menu_df = load_csv("menu.csv")
waste_df = load_csv("food-wastage.csv")

# PROCESS
menu_df = process_menu(menu_df)
menu_df = map_day_to_date(menu_df)

df = merge_data(menu_df, waste_df)
df = create_text_feature(df)

# TRAIN MODEL
model = WasteModel()
model.train(df)


def predict_waste(b, l, s, d):
    pred = model.predict(b, l, s, d)
    insight = model.get_insight(pred)
    return pred, insight


with gr.Blocks(title="Smart Mess Optimization System") as app:

    with gr.Tab("Prediction"):
        b = gr.Textbox(label="Breakfast")
        l = gr.Textbox(label="Lunch")
        s = gr.Textbox(label="Snacks")
        d = gr.Textbox(label="Dinner")

        out1 = gr.Number(label="Predicted Waste")
        out2 = gr.Textbox(label="Insight")

        btn = gr.Button("Predict")
        btn.click(predict_waste, inputs=[b, l, s, d], outputs=[out1, out2])

    with gr.Tab("Trends"):
        gr.Plot(model.get_trend_plot)

    with gr.Tab("Meal Analysis"):
        gr.Plot(model.get_meal_plot)

    with gr.Tab("Menu Impact"):
        gr.Plot(model.get_food_impact)


app.launch(server_name="0.0.0.0", server_port=7860)