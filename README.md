## Hospital Emergency Preparedness Capstone Project

### Executive Summary

#### The Problem area
The COVID pandemic exposed systemic challenges in hospitals related to staffing, bed availability, and supply shortages during infectious outbreaks. If hospitals had clear indicators of rising demand in advance, they could better address these issues, care for more patients, and deliver more comprehensive care during crises. This project aims to develop data-driven models that identify leading indications for hospital staffing shortages by U.S. region.

#### The User
Hospitals can benefit from the analyses and predictions from this project to ensure they are fully prepared for staffing shortages during small or large scale outbreaks. This can look ensuring more doctors and nurses are available and optimizing schedules based on the predictions of a model.

#### The Big Idea
Using machine learning methods such as Linear Regression, XGBoost, and Neural Networks, optimal models for a given U.S. region can be used to predict the number of hospitals that experience a shortage. The techniques used on a per region basis can be applied to individual hospitals and their staffing supply.

#### The Impact
Hospitals are known for being underfunded, and they require the support of a better public health system. This project is by no means a solution to the underlying public health infrastructure issues that hospitals face, but it can provide an example of how machine learning techniques can be used to predict staffing shortages. As an open source project, it can be extended by the community to provide more sophisticated approaches to health care forcasting and modeling.

#### The Data
A key dataset for this project is the COVID-19 Reported Patient Impact and Hospital Capacity by State Time Series data provided by [HealthData.gov](https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/g62h-syeh/about_data). It is a rich dataset with about 81,000 rows of time series data by state. It covers data such as the number of hospitals that have staffing shortages, number of hospitals that anticipate shortages, the number of inpatient beds available and utilized, admissions by age group, suspected COVID cases, percentage of COVID cases, deaths, and flu cases.

### Streamlit Application
The live application can be viewed on [Streamlit Cloud](https://evmiguel-hospital-shortage.streamlit.app/). It provides a visualization tool for the predictions of the three models and toggles for dates and lags.

### Local Development

1. Download data from [here](https://drive.google.com/drive/folders/1eWaBiZ5lzmiiJq-Ggaufb2A4R1Mz0RC0?usp=drive_link) into the `data` folder.
2. Activate the conda environment
```
conda env create -f conda.yml
conda activate capstone
```
3. Run the app
```
streamlit run src/streamlit/Visualization.py
```


### Methodology

1. Data cleaning. See `notebooks/01-data-loading-cleaning.ipynb`
2. Data preprocessing. See `notebooks/04-modelling.ipynb`
    - Features extracted using a lasso regression. See `get_features_by_region` in `lib/df_processing.py`
    - Create lagged DataFrames. See `create_rolling_df` in `lib/df_processing.py`
3. Modelling. See `notebooks/04-modelling.ipynb`


### Organization

#### Repository 

* `data` 
    - contains link to copy of the dataset (stored in a publicly accessible cloud storage)

* `model`
    - `joblib` dump of final model(s)

* `notebooks`
    - contains all final notebooks involved in the project

* `docs`
    - contains final report, presentations which summarize the project

* `references`
    - contains papers / tutorials used in the project

* `lib`
    - Contains the project code utilities

* `src`
    - Contains the project source code (refactored from the notebooks)

* `.gitignore`
    - Part of Git, includes files and folders to be ignored by Git version control

* `conda.yml`
    - Conda environment specification

* `README.md`
    - Project landing page (this page)

* `LICENSE`
    - Project license

#### Dataset

The dataset was sourced from [HealthData.gov](https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/g62h-syeh/about_data) on July 25, 2024.

