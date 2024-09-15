import numpy as np
import pandas as pd
import streamlit as st
import joblib
import json
import torch
import torch.nn as nn
import sys; sys.path.insert(0, '.')
from lib import df_processing

#######################################################################################
# CONFIGURATION
#######################################################################################

st.set_page_config(page_title="COVID Staff Shortage Predictor")

#######################################################################################
# NEURAL NET FOR LOADING MODEL
#######################################################################################

class SimpleNN(nn.Module):
    """Basic multi-layer architecture."""

    def __init__(self, columns = 1):
        """Define the main components of the network"""
        super(SimpleNN, self).__init__()

        self.layer_1 = nn.Linear(columns, 32) # transition from input into hidden layer
        self.activation_1 = nn.ReLU()   # Activation function
        self.layer_2 = nn.Linear(32, 16)  # transition from hidden layer into output
        self.activation_2 = nn.ReLU()
        self.layer_3 = nn.Linear(16, 8)
        self.activation_3 = nn.ReLU()
        self.layer_4 = nn.Linear(8, 2)

        # Declare a regression layer
        self.regression_layer = nn.Linear(2,1)

    def forward(self, x):
        """Perform forward pass."""

        # pass through the layers
        hidden_1 = self.activation_1(self.layer_1(x))
        hidden_2 = self.layer_2(hidden_1)
        hidden_3 = self.layer_3(self.activation_2(hidden_2))
        hidden_4 = self.layer_4(self.activation_3(hidden_3))

        # Notice the network will behave differently based on
        # whether it is training or not
        output = self.regression_layer(hidden_4)

        # return output
        return output

#######################################################################################
# DATA LOADING
#######################################################################################

new_england_df = pd.read_csv("./data/cleaned_new_england_covid.csv")
mid_atlantic_df = pd.read_csv("./data/cleaned_mid_atlantic_covid.csv")
south_df = pd.read_csv("./data/cleaned_south_covid.csv")
midwest_df = pd.read_csv("./data/cleaned_midwest_covid.csv")
southwest_df = pd.read_csv("./data/cleaned_southwest_covid.csv")
west_df = pd.read_csv("./data/cleaned_west_covid.csv")

df_processing.set_indexes([new_england_df, mid_atlantic_df, south_df, midwest_df, southwest_df, west_df])

dfs_region_map = {
    "New England": new_england_df.groupby("date").mean(numeric_only=True),
    "Mid Atlantic": mid_atlantic_df.groupby("date").mean(numeric_only=True),
    "South": south_df.groupby("date").mean(numeric_only=True),
    "Midwest": midwest_df.groupby("date").mean(numeric_only=True),
    "Southwest": southwest_df.groupby("date").mean(numeric_only=True),
    "West": west_df.groupby("date").mean(numeric_only=True)
}

with open("./data/r2_pkl_lr.json", 'r') as file:
    r2_pkl_lr = json.load(file)

with open("./data/r2_pkl_xgb.json", 'r') as file:
    r2_pkl_xgb = json.load(file)

with open("./data/r2_pkl_nn.json", 'r') as file:
    r2_pkl_nn = json.load(file)

best_model = st.empty()
r2 = st.empty()
prediction = st.empty()

#######################################################################################
# APPLICATION
#######################################################################################

region = st.sidebar.selectbox("Select a region", ["New England", "Mid Atlantic", "Midwest", "South", "Southwest", "West"])

model_type = st.sidebar.selectbox("Select a model", ["Linear Regression", "XGBoost", "Neural Net"])


if model_type == "Linear Regression":
    best_model.header("Linear Regression")
    r2.text(f"R^2: {r2_pkl_lr[region]["r2"] * 100:.2f}%")
    model = joblib.load(f"models/{r2_pkl_lr[region]["filename"]}")
    raw_features = [feature.split("-")[0] if len(feature.split("-")) == 2 else feature.split("-")[0] + "-" + feature.split("-")[1] for feature in r2_pkl_lr[region]["features"]]
elif model_type == "XGBoost":
    best_model.header("XGBoost")
    r2.text(f"R^2: {r2_pkl_xgb[region]["r2"] * 100:.2f}%")
    model = joblib.load(f"models/{r2_pkl_xgb[region]["filename"]}")
    raw_features = [feature.split("-")[0] if len(feature.split("-")) == 2 else feature.split("-")[0] + "-" + feature.split("-")[1] for feature in r2_pkl_xgb[region]["features"]]
else:
    best_model.header("Neural Net")
    r2.text(f"R^2: {r2_pkl_nn[region]["r2"] * 100:.2f}%")
    raw_features = [feature.split("-")[0] if len(feature.split("-")) == 2 else feature.split("-")[0] + "-" + feature.split("-")[1] for feature in r2_pkl_nn[region]["features"]]

feature_map = {}
for feature in raw_features:
    feature_map[feature] = st.sidebar.slider(feature, dfs_region_map[region][feature].min(), dfs_region_map[region][feature].max())

input_df = pd.DataFrame(feature_map, index=[0])
if model_type == "Neural Net":
    model = torch.load(f"models/{r2_pkl_nn[region]["filename"]}", weights_only=False)
    prediction.text(model(torch.tensor(input_df.values, dtype=torch.float32)).item())
else:
    prediction.text(model.predict(input_df)[0])


