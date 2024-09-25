import numpy as np
import pandas as pd
import streamlit as st
import joblib
import json
import torch
import torch.nn as nn
import sys; sys.path.insert(0, '.')
from lib import df_processing
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, r2_score
import datetime

#######################################################################################
# CONFIGURATION
#######################################################################################

st.set_page_config(page_title="COVID Staff Shortage Predictor", layout="wide")

def get_feature_name(feature):
    return feature.split("-")[0] if len(feature.split("-")) == 2 else feature.split("-")[0] + "-" + feature.split("-")[1]

def highlight_max(s, df):
    name = s.name
    cells = []
    if name == "Critical Staffing Shortage Today Yes":
        second = np.argsort(df.T)[1]
        for i, cell in enumerate(s):
            if i == second:
                cells.append("background-color: green")
            elif i == 0:
                cells.append("background-color: orange")
            else:
                cells.append(None)
        return cells
    return [None, None, None, None]

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

lag = 1

#######################################################################################
# APPLICATION
#######################################################################################

region = st.sidebar.selectbox("Select a region", ["New England", "Mid Atlantic", "Midwest", "South", "Southwest", "West"])


if region == "New England":
    options = [datetime.datetime(2021, 1, 1), datetime.datetime(2021, 7, 1), datetime.datetime(2022, 1, 1), datetime.datetime(2024, 4, 1)]
elif region == "Mid Atlantic":
    options = [datetime.datetime(2020, 12, 15), datetime.datetime(2021, 3, 1), datetime.datetime(2021, 12, 1), datetime.datetime(2022, 5, 1), datetime.datetime(2022, 8, 1), datetime.datetime(2024, 4, 1)]
elif region == "Midwest":
    options = [datetime.datetime(2020, 12, 1), datetime.datetime(2021, 7, 1), datetime.datetime(2022, 1, 1), datetime.datetime(2024, 4, 1)]
elif region == "South":
    options = [datetime.datetime(2021, 1, 1), datetime.datetime(2021, 7, 1), datetime.datetime(2021, 9, 1), datetime.datetime(2024, 4, 1)]
elif region == "Southwest":
    options = [datetime.datetime(2020, 8, 1), datetime.datetime(2021, 1, 1), datetime.datetime(2021, 7, 1), datetime.datetime(2022, 1, 1), datetime.datetime(2024, 4, 1)]
else:
    options = [datetime.datetime(2020, 7, 1), datetime.datetime(2021, 1, 1), datetime.datetime(2021, 7, 1), datetime.datetime(2022, 1, 1), datetime.datetime(2024, 4, 1)]

chosen_date = st.sidebar.select_slider("Select a date", options=options)

min_default_date = dfs_region_map[region].index.min().to_pydatetime()
max_default_date = dfs_region_map[region].index.max().to_pydatetime()

lag = st.sidebar.select_slider("Lag", options=[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
lag_text = st.header(f"Lag: {lag}")
rolling_df = df_processing.create_rolling_df(dfs_region_map[region], lag, ignore_columns=["critical_staffing_shortage_today_yes"])
actual_shortage = dfs_region_map[region].loc[dfs_region_map[region].index == chosen_date, "critical_staffing_shortage_today_yes"]

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
# Dataframes
##################################################################
shortages = [round(actual_shortage.values[0], 2), round(shortage_lr, 2), round(shortage_xgb, 2), round(shortage_nn, 2)]
shortages_df = pd.DataFrame(data=shortages, index=["Actual", "Linear Regression", "XGBoost", "Neural Net"], columns=["Critical Staffing Shortage Today Yes"])
errors = [mean_absolute_error(actual_shortage, actual_shortage), mean_absolute_error(actual_shortage, [shortage_lr]), mean_absolute_error(actual_shortage, [shortage_xgb]), mean_absolute_error(actual_shortage, [shortage_nn])]
errors_df = pd.DataFrame(data=errors, index=["Actual", "Linear Regression", "XGBoost", "Neural Net"], columns=["Mean Absolute Error"])
top_df = pd.concat([shortages_df, errors_df], axis=1)
st.dataframe(top_df.T.style.apply(highlight_max, axis=1, df=top_df.T.loc["Mean Absolute Error"]))

##################################################################
# Plotting
##################################################################
fig = go.Figure()
line_graph = go.Line(x=pd.date_range(min_default_date, max_default_date), y=dfs_region_map[region].loc[:, "critical_staffing_shortage_today_yes"], name="critical staffing shortage over time", opacity=0.5)
actual_point = go.Scatter(x=[chosen_date], y=actual_shortage, marker_symbol="x", marker_size=15, marker_color="#1BFC06", name="actual shortage value")
lr_point = go.Scatter(x=[chosen_date], y=[shortage_lr], marker_symbol="circle", marker_color="red", marker_size=10, name="Linear regression prediction")
xgb_point = go.Scatter(x=[chosen_date], y=[shortage_xgb], marker_symbol="star", marker_color="blue", marker_size=10, name="XGB prediction")
nn_point = go.Scatter(x=[chosen_date], y=[shortage_nn], marker_symbol="triangle-up", marker_color="orange", marker_size=10, name="Neural net prediction")
fig.add_trace(line_graph)
fig.add_trace(actual_point)
fig.add_trace(lr_point)
fig.add_trace(xgb_point)
fig.add_trace(nn_point)
fig.update_layout(height=600, legend={
            "x": 0.8,
            "y": 0.9})
st.plotly_chart(fig, use_container_width=True)





