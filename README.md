## Hospital Emergency Preparedness Capstone Project

### Executive Summary

#### The Problem area
The COVID pandemic exposed systemic challenges in hospitals related to staffing, bed availability, and supply shortages during infectious outbreaks. If hospitals had clear indicators of rising demand in advance, they could better address these issues, care for more patients, and deliver more comprehensive care during crises. This project aims to develop data-driven models that identify leading indications, enabling the implementation of effective policies and procedures.

#### The User
Hospitals, nursing homes, prisons, and urgent care can benefit from the analyses and predictions from this project to ensure they are fully prepared for the small or large scale outbreaks. They can use the tools developed from this project to be more proactive in their approach to outbreaks in their area. This can look like preparing more beds for patients and ensuring more doctors and nurses are available for these situations. Facilities can better allocate the resources that they already have.

#### The Big Idea
Given the dataset below, there are many ways to use machine learning to create solutions for hospitals. Machine learning could be used to predict the number of beds needed given the previous day's admission and number of people who were positive with COVID. It can also be used to estimate the number of staff needed based on whether there was a shortage the previous day. Machine learning can also help predict the rate of the virus based on the admissions in a given time frame.

#### The Impact
Hospitals are known for being underfunded, and they require the support of a better public health system3. This project is by no means a solution to the underlying public health infrastructure issues that hospitals face, but it can ease them and other related facilities when handling outbreaks in their area. This project will most likely not increase hospital revenue, but it will improve efficiency in staffing and providing supplies to admitted patients. It will also help isolate more complex patients, suffering from the flu and COVID for example, and move them to separate units for better care.

#### The Data
A key dataset for this project is the COVID-19 Reported Patient Impact and Hospital Capacity by State Time Series data provided by [HealthData.gov](https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/g62h-syeh/about_data). It is a rich dataset with about 81,000 rows of time series data by state. It covers data such as critical staff shortages, anticipated shortages, the number of inpatient beds available and utilized, admissions by age group, suspected COVID cases, percentage of COVID cases, deaths, and flu cases.

### Demo

TBD


### Methodology

TBD


### Organization

#### Repository 

* `data` 
    - contains link to copy of the dataset (stored in a publicly accessible cloud storage)
    - saved copy of aggregated / processed data as long as those are not too large (> 10 MB)

* `model`
    - `joblib` dump of final model(s)

* `notebooks`
    - contains all final notebooks involved in the project

* `docs`
    - contains final report, presentations which summarize the project

* `references`
    - contains papers / tutorials used in the project

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

