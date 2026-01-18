import gradio as gr
import pandas as pd
import numpy as np
import pickle

with open("water_level.pkl", "rb") as f:
    model = pickle.load(f)

def predict_potability(ph, Hardness, Solids, Chloramines, Sulfate,
                       Conductivity, Organic_carbon, Trihalomethanes, Turbidity):

    input_df = pd.DataFrame([[
        ph, Hardness, Solids, Chloramines, Sulfate,
        Conductivity, Organic_carbon, Trihalomethanes, Turbidity
    ]],
    columns=[
        'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
    ])

    input_df = input_df.fillna(input_df.median())
    pred = model.predict(input_df)[0]
    return f"Predicted Potability: {pred}"

inputs = [
    gr.Number(label="pH"),
    gr.Number(label="Hardness"),
    gr.Number(label="Solids"),
    gr.Number(label="Chloramines"),
    gr.Number(label="Sulfate"),
    gr.Number(label="Conductivity"),
    gr.Number(label="Organic Carbon"),
    gr.Number(label="Trihalomethanes"),
    gr.Number(label="Turbidity")
]

app = gr.Interface(
    fn=predict_potability,
    inputs=inputs,
    outputs="text",
    title="Water Potability Predictor"
)

app.launch(share=True)