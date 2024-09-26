import streamlit as st

st.markdown("## Background")
st.write("""This project was inspired by my mother, a nurse, who worked the beginning of her career during the AIDS epidemic and the latter part
            of her career during the COVID pandemic. She recounted that during COVID, she was one of a few nurses on her unit, taking care of sick patients who
            were not only fighting COVID but also suffering from the lack of care due to staffing shortages, specifically nurses and doctors. I, too, was hospitalized
            during the pandemic, and I remember the chaos of being in a hospital during that time. My hope for this project is to use data science tools to provide a way
            for hospitals to potentially alleviate the issue of staffing shortages so that both patients and staff can have a better experience.""")

st.markdown("## Data")
st.write("""The raw data for this project can be found on [healthdata.gov](https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/g62h-syeh/about_data). 
            It was accessed on July 25, 2024. It is a rich dataset with about 81,000 rows of time series data by state. It covers data such as critical staff shortages, anticipated shortages, 
            the number of inpatient beds available and utilized, admissions by age group, suspected COVID cases, percentage of COVID cases, deaths, and flu cases.""")

st.markdown("## Preprocessing")
st.markdown("""
    1. Pediatric data was dropped because children only accounted for 0.1% of COVID deaths in the country<sup>[1](https://med.stanford.edu/news/all-news/2023/10/children-covid.html)</sup>
    2. Missing values were imputed using a forward fill and a backward fill
    3. States were grouped into regions for easier processing
    4. Lasso regression was run to select features for each region
""", unsafe_allow_html=True)
st.markdown("""See [this notebook](https://github.com/evmiguel/covid-brainstation-capstone/blob/main/notebooks/01-data-loading-cleaning.ipynb) for steps 1-3, and the __get_features_by_region__ [function](https://github.com/evmiguel/covid-brainstation-capstone/blob/9939612cd55a98033fb5e7ed1a7a1eb9c27abfbb/lib/df_processing.py#L28)
    for step 4.""")

st.markdown("## Modeling")
st.markdown("""
    After selecting the features for each region, 100 dataframes were created for each region. Each new dataframe contained a lag of *t - n*, where n is a number between 1 and 100.
    The intermediary lags between *t* and *t - n* were dropped to observe how each lagged feature at a given n would perform against the following models:
""")
st.markdown("""
    - Linear Regression
    - XGBoost
    - Neural Network
""")
st.markdown("""The goal of each model was to predict the variable __critical_staffing_shortage_today_yes__.""") 
st.markdown("""See [this notebook](https://github.com/evmiguel/covid-brainstation-capstone/blob/main/notebooks/04-modelling.ipynb) 
            on how the models were created using the lagged dataframes. The function to create the lagged dataframes is [__create_rolling_df__](https://github.com/evmiguel/covid-brainstation-capstone/blob/9939612cd55a98033fb5e7ed1a7a1eb9c27abfbb/lib/df_processing.py#L78).""")

st.markdown("## Mid Atlantic Case Study")
st.markdown("This section examines the findings for the hospital staffing shortages of the Mid Atlantic region and can be applied to the other regions.")

st.markdown("### Baseline Model")
st.write("""The baseline model for predicting critical staffing shortage is the mean. As we can see, the mean intersects the graph at some points, but it does not cover most of the points.
            The following models show the best model on the left and the worst model on the right relative to the number of lags.
            """)
st.image('img/mean_model_mid_atlantic.png')

st.markdown("### Linear Regression Model")
st.markdown("""The best model for linear regression had a lag of 1 with an R<sup>2</sup> of 92%, and the worst model had a lag of 73 with an R<sup>2</sup> of 56%. 
            (See [this notebook](https://github.com/evmiguel/covid-brainstation-capstone/blob/main/notebooks/04-modelling.ipynb) for R<sup>2</sup> calculations.)
            The best model predicts the staffing shortage well relative to the mean. While the worst model also performs better than the mean, we can observe that
            the higher lag leads to more inaccurate predictions.
            """, unsafe_allow_html=True)
st.image('img/lr_model_mid_atlantic.png')

st.markdown("### XGBoost Model")
st.markdown("""The best model for XGBoost had a lag of 10 with an R<sup>2</sup> of 97%, and the worst model had a lag of 46 with an R<sup>2</sup> of 85%. 
            (See [this notebook](https://github.com/evmiguel/covid-brainstation-capstone/blob/main/notebooks/04-modelling.ipynb) for R<sup>2</sup> calculations.)
            Compared to the linear regression model, XGBoost's best model performs better and can predict hospital staffing shortages at higher lags with good R<sup>2</sup> scores, 
            even with its worst model.
            """, unsafe_allow_html=True)
st.image('img/xgb_model_mid_atlantic.png')

st.markdown("### Neural Network")
st.markdown("""The best model for the neural network had a lag of 1 with an R<sup>2</sup> of 96%, and the worst model had a lag of 74 with an R<sup>2</sup> of 85%. 
            (See [this notebook](https://github.com/evmiguel/covid-brainstation-capstone/blob/main/notebooks/04-modelling.ipynb) for R<sup>2</sup> calculations.)
            The neural network performs somewhere in between the linear regression model and XGBoost because its best model only has a lag of 1, but its worst model
            has the highest lag with the same R<sup>2</sup> as XGBoost's worst model.
            """, unsafe_allow_html=True)
st.image('img/nn_model_mid_atlantic.png')

st.markdown("### R<sup>2</sup> Analysis", unsafe_allow_html=True)
st.markdown("""As shown in the R<sup>2</sup> graphs, linear regression is sensitive to higher lags, as it falls below the 90% cutoff at close to 0 lags.
            XGBoost and neural net are better at explaining the variance in the data with higher lags and are therefore more tolerant of predicting hospital
            shortages going further into the past. XGBoost has the greatest tolerance with a lag of 30 before crossing the cutoff. It is important to note that
            linear regression provides better interpretability because its graph does not fluctuate as much as the other two graphs.
""", unsafe_allow_html=True)
st.image('img/r2_complete_mid_atlantic.png')

st.markdown("### Mean Absolute Error Analysis")
st.write("""As shown in the Mean Absolute Error graphs, like the R<sup>2</sup> graphs, linear regression is sensitive to higher lags, as the mean absolute error (MAE)
            increases steadily with higher lags. Note that the range and magnitude of errors decrease with XGBoost and the neural network, but their graphs fluctuate as
            the number of lags increases. Moreover, XGBoost has the smallest errors.""")
st.image('img/error_complete_mid_atlantic.png')

st.markdown("### Conclusion")
st.markdown("""XGBoost is the best performing model at predicting hospital staffing shortages because of its flexibility in the number of lags producing high R<sup>2</sup>
            values. However, because it does not offer the same level of interpretability that linear regression does, linear regression is suitable in instances where the model
            needs to be easily explained, especially when extracting feature importances. The neural net, while still useful, does not offer the flexibility that XGBoost provides
            nor the interpretability that linear regression gives and should, therefore, be used with caution.""", unsafe_allow_html=True)

st.markdown("*This project is maintained by [Erika Miguel](https://erikamiguel.com).*")
