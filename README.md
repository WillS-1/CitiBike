# CitiBike Demand Prediction Model in R

This repository contains an R-based machine learning model built to predict bike demand for CitiBike stations using **XGBoost**. The model considers factors such as station location, time of day, weather conditions, and projected monthly revenue to forecast the number of bikes required at each station.

## Overview

CitiBike is a bike-sharing program operating in various cities. Predicting demand for bikes at each station is crucial for optimizing fleet distribution, improving operational efficiency, and ensuring customers can easily find available bikes. This model uses **XGBoost**, a popular gradient boosting algorithm, to predict the number of bikes needed at each station at a given time.

The model leverages multiple features such as:
- Station location and usage patterns
- Time of day and day of the week
- Member type (casual rider or member)
- Bike Type (Classic or electric)

## Data

The dataset used for this project includes historical bike trip data from CitiBike stations, along with weather information, station details, and projected revenue. The dataset contains the following key columns:
- `station_id`: Unique identifier for each CitiBike station
- `datetime`: Date and time of the trip
- `bikes_rented`: Number of bikes rented at a station at a given time
- `location`: Latitude and longitude of the station
- `member_type`: Type of rider if the are a member or just casual rider

## Model

The model is built using **XGBoost**, a powerful implementation of gradient boosting. The key steps in the model development include:
1. Data preprocessing: Cleaning and transforming the data, including handling missing values and scaling numerical features.
2. Feature engineering: Creating new features like time of day, day of the week and ride distance.
3. Model training: Using the XGBoost algorithm to train the model on historical bike trip data.
4. Hyperparameter tuning: Optimizing the model’s hyperparameters to improve predictive performance.

The full model is available in this repository under `CitiBike Machine Learning Project.R`

### Key Libraries Used:
- `geosphere`: For spatial calculations (e.g., distance between stations)
- `ggplot2`: For data visualization
- `caret`: For model training and evaluation
- `xgboost`: For the gradient boosting model
- `lubridate`: For date-time manipulation
- `sf`: For handling spatial data (stations' locations)
- `scales`: For better visualization of scales in plots
- `stringr`: For string manipulation
- `dplyr`: For data manipulation and transformations

## Conclusion:
I have also included a document detailing my conclusions and reccomendations based on the outcome of this project. 

## Acknowledgements
Thanks to CitiBike for providing this data for the analysis
