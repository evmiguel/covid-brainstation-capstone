import numpy as np
import pandas as pd
import streamlit as st
import joblib
import json
import torch
import torch.nn as nn
import sys; sys.path.insert(0, '.')
from lib import df_processing
import matplotlib.pyplot as plt
import datetime

#######################################################################################
# CONFIGURATION
#######################################################################################

st.set_page_config(page_title="COVID Staff Shortage Predictor")

def get_feature_name(feature):
    return feature.split("-")[0] if len(feature.split("-")) == 2 else feature.split("-")[0] + "-" + feature.split("-")[1]

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
prediction_lr = st.empty()
prediction_xgb = st.empty()
prediction_nn = st.empty()
lag = 1

#######################################################################################
# APPLICATION
#######################################################################################

region = st.sidebar.selectbox("Select a region", ["New England", "Mid Atlantic", "Midwest", "South", "Southwest", "West"])


if region == "New England":
    options = [datetime.datetime(2021, 1, 1), datetime.datetime(2021, 7, 1), datetime.datetime(2024, 4, 1)]
elif region == "Mid Atlantic":
    options = [datetime.datetime(2020, 11, 1), datetime.datetime(2020, 12, 1), datetime.datetime(2021, 12, 1), datetime.datetime(2022, 5, 1), datetime.datetime(2022, 8, 1), datetime.datetime(2024, 4, 1)]
elif region == "Midwest":
    options = [datetime.datetime(2020, 8, 1), datetime.datetime(2021, 7, 1), datetime.datetime(2022, 2, 1), datetime.datetime(2024, 4, 1)]
elif region == "South":
    options = [datetime.datetime(2021, 1, 1), datetime.datetime(2021, 7, 1), datetime.datetime(2022, 1, 1), datetime.datetime(2024, 4, 1)]
elif region == "Southwest":
    options = [datetime.datetime(2020, 8, 1), datetime.datetime(2021, 1, 1), datetime.datetime(2021, 7, 1), datetime.datetime(2022, 1, 1), datetime.datetime(2024, 4, 1)]
else:
    options = [datetime.datetime(2020, 7, 1), datetime.datetime(2021, 1, 1), datetime.datetime(2021, 7, 1), datetime.datetime(2022, 1, 1), datetime.datetime(2024, 4, 1)]

chosen_date = st.sidebar.select_slider("Select a date", options=options)

min_default_date = dfs_region_map[region].index.min().to_pydatetime()

lag = st.sidebar.select_slider("Lag", options=[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
rolling_df = df_processing.create_rolling_df(dfs_region_map[region], lag, ignore_columns=["critical_staffing_shortage_today_yes"])

##################################################################
# LINEAR REGRESSION
##################################################################
model = joblib.load(f"models/lr-{lag}_{region.lower().replace(" ", "_")}.pkl")
preprocessor = joblib.load(f"models/preprocessor-lr-{lag}_{region.lower().replace(" ", "_")}.pkl")
raw_features = [feature.split("-")[0] if len(feature.split("-")) == 2 else feature.split("-")[0] + "-" + feature.split("-")[1] for feature in r2_pkl_lr[region][0]["features"]]
columns = [column for column in rolling_df.columns.drop("critical_staffing_shortage_today_yes") if get_feature_name(column) in raw_features]
input_df = rolling_df.loc[rolling_df.index == chosen_date, columns]
scaled_input = preprocessor.transform(input_df)
shortage_lr = model.predict(scaled_input)[0]

##################################################################
# XGBoost
##################################################################
model = joblib.load(f"models/xgb-{lag}_{region.lower().replace(" ", "_")}.pkl")
preprocessor = joblib.load(f"models/preprocessor-xgb-{lag}_{region.lower().replace(" ", "_")}.pkl")
raw_features = [feature.split("-")[0] if len(feature.split("-")) == 2 else feature.split("-")[0] + "-" + feature.split("-")[1] for feature in r2_pkl_xgb[region][0]["features"]]
columns = [column for column in rolling_df.columns.drop("critical_staffing_shortage_today_yes") if get_feature_name(column) in raw_features]
input_df = rolling_df.loc[rolling_df.index == chosen_date, columns]
scaled_input = preprocessor.transform(input_df)
shortage_xgb = model.predict(scaled_input)[0]

##################################################################
# Neural Net
##################################################################
model = torch.load(f"models/nn-{lag}_{region.lower().replace(" ", "_")}.pt", weights_only=False)
preprocessor = joblib.load(f"models/preprocessor-nn-{lag}_{region.lower().replace(" ", "_")}.pt")
raw_features = [feature.split("-")[0] if len(feature.split("-")) == 2 else feature.split("-")[0] + "-" + feature.split("-")[1] for feature in r2_pkl_nn[region][0]["features"]]
columns = [column for column in rolling_df.columns.drop("critical_staffing_shortage_today_yes") if get_feature_name(column) in raw_features]
input_df = rolling_df.loc[rolling_df.index == chosen_date, columns]
scaled_input = preprocessor.transform(input_df)
shortage_nn = model(torch.tensor(scaled_input, dtype=torch.float32)).item()


##################################################################
# Plotting
##################################################################
fig, ax = plt.subplots()
ax.set_xlim(min_default_date - datetime.timedelta(days=60), chosen_date + datetime.timedelta(days=60))
ax.plot(dfs_region_map[region].loc[dfs_region_map[region].index <= chosen_date, "critical_staffing_shortage_today_yes"], alpha=0.2)
ax.plot(chosen_date, dfs_region_map[region].loc[dfs_region_map[region].index == chosen_date, "critical_staffing_shortage_today_yes"], "go")
ax.plot(chosen_date, shortage_lr, color="red", marker="o")
ax.plot(chosen_date, shortage_xgb, color="blue", marker="*")
ax.plot(chosen_date, shortage_xgb, color="orange", marker="X")
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
st.pyplot(fig)

prediction_lr.text(f"Staffing shortage Linear Regression: {round(shortage_lr)}")
prediction_xgb.text(f"Staffing shortage XGBoost: {round(shortage_xgb)}")
prediction_nn.text(f"Staffing shortage Neural Net: {round(shortage_nn)}")





