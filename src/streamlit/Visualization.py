import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import torch
import torch.nn as nn
import sys; sys.path.insert(0, '.')
from lib import df_processing
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error, r2_score
import datetime
st.set_page_config(page_title="COVID Staff Shortage Predictor", layout="wide")
st.sidebar.title("Predict Regional Hospital Staffing Shortage :stethoscope:")


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
# CONFIGURATION
#######################################################################################

st.markdown("""
<style>
    [data-testid="stMetric"] {
        text-align: center;
        padding: 15px 0;
    }   

    [data-testid="stMetricLabel"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    [data-testid="stMetricDeltaIcon-Up"] {
        display: none
    }
</style>
""", unsafe_allow_html=True)

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

regions = {
    "New England": {
        "states": ["ME", "NH", "VT", "MA", "CT", "RI"],
        "center": {
            "lat": 43.0995,
            "lon": -71.5905
        }
    },
    "Mid Atlantic": {
        "states": ["NY", "PA", "NJ", "MD", "DE", "DC"],
        "center": {
            "lat": 40.1063,
            "lon": -76.1596
        }
    },
    "Midwest": {
        "states": ["ND", "SD", "NE", "KS", "MN", "IA", "MO", "WI", "IL", "MI", "IN", "OH"],
        "center": {
            "lat": 42.3,
            "lon": -92.675
        }
    },
    "South": {
        "states": ["WV", "VA", "NC", "SC", "GA", "FL", "KY", "TN", "AL", "MS", "AR", "LA"],
        "center": {
            "lat": 34.1288,
            "lon": -84.6132
        }
    }
    ,
    "Southwest": {
        "states": ["AZ", "NM", "TX", "OK"],
        "center": {
            "lat": 33.95,
            "lon": -103.65
        }
    },
    "West": {
        "states": ["WA", "OR", "CA", "ID", "NV", "UT", "MT", "WY", "CO", "AK", "HI"],
        "center": {
            "lat": 37.95,
            "lon": -103.65
        }
    }
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
    options = [datetime.date(2021, 1, 1), datetime.date(2021, 7, 1), datetime.date(2022, 1, 1), datetime.date(2024, 4, 1)]
elif region == "Mid Atlantic":
    options = [datetime.date(2020, 12, 15), datetime.date(2021, 3, 1), datetime.date(2021, 12, 1), datetime.date(2022, 5, 1), datetime.date(2022, 8, 1), datetime.date(2024, 4, 1)]
elif region == "Midwest":
    options = [datetime.date(2020, 12, 1), datetime.date(2021, 7, 1), datetime.date(2022, 1, 1), datetime.date(2024, 4, 1)]
elif region == "South":
    options = [datetime.date(2021, 1, 1), datetime.date(2021, 7, 1), datetime.date(2021, 9, 1), datetime.date(2024, 4, 1)]
elif region == "Southwest":
    options = [datetime.date(2020, 8, 1), datetime.date(2021, 1, 1), datetime.date(2021, 7, 1), datetime.date(2022, 1, 1), datetime.date(2024, 4, 1)]
else:
    options = [datetime.date(2020, 7, 1), datetime.date(2021, 1, 1), datetime.date(2021, 7, 1), datetime.date(2022, 1, 1), datetime.date(2024, 4, 1)]

chosen_date = datetime.datetime.combine(st.sidebar.selectbox("Select a date", options=options), datetime.time.min)

min_default_date = dfs_region_map[region].index.min().to_pydatetime()
max_default_date = dfs_region_map[region].index.max().to_pydatetime()

lag = st.sidebar.select_slider("Select a Lag", options=[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
st.sidebar.write("*What are lag features?*")
st.sidebar.write("Lag features are values at prior time steps. For example, a lag of 1 means that the features are pulled from *t-1*.")
st.sidebar.write("*How are lagged features used in this app?*")
st.sidebar.write("We run different models that use the lags from 1 to 100 to predict the number of hospitals experiencing staffing shortages and see how those models compare.")

rolling_df = df_processing.create_rolling_df(dfs_region_map[region], lag, ignore_columns=["critical_staffing_shortage_today_yes"])
actual_shortage = dfs_region_map[region].loc[dfs_region_map[region].index == chosen_date, "critical_staffing_shortage_today_yes"]

col1, col2, col3 = st.columns((1.5, 4.5, 2), gap='medium')

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

with col1:
    st.markdown('#### Number of Hospitals with Staffing Shortages')
    st.metric(label="Actual", value=f"{actual_shortage.values[0]:.2f}")
    st.metric(label="Linear Regression", value=f"{shortage_lr:.2f}", delta=f"{round(mean_absolute_error(actual_shortage, [shortage_lr]), 2)} MAE", delta_color="off")
    st.metric(label="XGBoost", value=f"{shortage_xgb:.2f}", delta=f"{round(mean_absolute_error(actual_shortage, [shortage_xgb]), 2)} MAE", delta_color="off")
    st.metric(label="Neural Net", value=f"{shortage_nn:.2f}", delta=f"{round(mean_absolute_error(actual_shortage, [shortage_nn]), 2)} MAE", delta_color="off")

##################################################################
# Plotting
##################################################################
with col2:
    fig = go.Figure()
    line_graph = go.Scatter(x=pd.date_range(min_default_date, max_default_date), y=dfs_region_map[region].loc[:, "critical_staffing_shortage_today_yes"], name="average critical staffing shortage over time", opacity=0.5)
    actual_point = go.Scatter(x=[chosen_date], y=actual_shortage, marker_symbol="x", marker_size=20, marker_color="#1BFC06", name="actual value")
    lr_point = go.Scatter(x=[chosen_date], y=[shortage_lr], marker_symbol="circle", marker_color="red", marker_size=15, name="Linear regression prediction")
    xgb_point = go.Scatter(x=[chosen_date], y=[shortage_xgb], marker_symbol="star", marker_color="blue", marker_size=15, name="XGB prediction")
    nn_point = go.Scatter(x=[chosen_date], y=[shortage_nn], marker_symbol="triangle-up", marker_color="orange", marker_size=15, name="Neural net prediction")
    fig.add_trace(line_graph)
    fig.add_trace(actual_point)
    fig.add_trace(lr_point)
    fig.add_trace(xgb_point)
    fig.add_trace(nn_point)
    fig.update_layout(height=600, legend={
                "x": 0.8,
                "y": 0.9},
                title="Critical Staffing Shortage Today Yes",
                title_x=0.2,
                xaxis_title="Date",
                yaxis_title="Number of Hospitals with Staffing Shortages")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""<div id="explanation"><i>Critical Staffing Shortage Today Yes</i> is the target variable. It represents the average number of hospitals reporting a critical staffing shortage in this region on a particular day.
                </div>""", unsafe_allow_html=True)

with col3:
    fitbounds = "locations" if region != "West" else False
    map = px.choropleth(title=region, locations=regions[region]["states"], locationmode="USA-states", center=regions[region]["center"], scope="usa", fitbounds=fitbounds)
    map.update_traces(hoverinfo="skip", hovertemplate=None)
    map.update_layout(showlegend=False, title=dict(y=.75))
    st.plotly_chart(map)

    st.markdown("""
            <div id="xgb_blurb">
            <img src="./app/static/xgb_x.png" width="25" height="25">
            <p>XGBoost tends to give the best predictions overall.</p>
            </div>
            <style>
                #xgb_blurb {
                    display: flex;
                    flex-direction: row;
                    align-items: center;
                }

                #xgb_blurb > img {
                    margin-right: 5px;
                }
            </style>
            """, unsafe_allow_html=True)
    
    st.markdown("""
                <div id="lr_blurb">
                <img src="./app/static/lr_c.png" width="25" height="25">
                <p>Linear Regression tends to exaggerate the prediction with large lags.</p>
                </div>
                <style>
                    #lr_blurb {
                        display: flex;
                        flex-direction: row;
                        align-items: center;
                    }

                    #lr_blurb > img {
                        margin-right: 5px;
                    }
                </style>
                """, unsafe_allow_html=True)
    
    st.markdown("""
                <div id="nn_blurb">
                <img src="./app/static/nn_t.png" width="25" height="25">
                <p>Neural net tends to give mixed results.</p>
                </div>
                <style>
                    #nn_blurb {
                        display: flex;
                        flex-direction: row;
                        align-items: center;
                    }

                    #nn_blurb > img {
                        margin-right: 5px;
                    }
                </style>
                """, unsafe_allow_html=True)

st.markdown("""<div id="maintainer"><i>This project is maintained by <a href="https://erikamiguel.com">Erika Miguel</a>.</i>
                </div><style>
                    #maintainer {
                        margin-top: 5em;
                        text-align: center;
                    }
                </style>""", unsafe_allow_html=True)