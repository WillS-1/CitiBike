knitr::opts_chunk$set(warning = FALSE, message = FALSE)

#'## NYC Citibike Demand Project
#' This project focuses on analyzing CitiBike station performance using geospatial data, temporal patterns, and machine learning techniques to predict demand and optimize bike allocation. The main objectives include:
#'
#' **1. Data Processing:**
#' 
#' The dataset includes ride details such as start and end coordinates, timestamps, and membership type. Distance between stations is calculated using the Haversine formula, day of the week and hour of the day are extracted.
#'
#' **2. Station Level Insights:**
#' 
#' Key metrics, including total rides, average distance, casual vs. member ride counts, and weekend activity, are computed for each station. Stations are grouped by clusters based on geographic location using K-means clustering.
#'
#' **3. Demand Prediction:**
#' 
#' Using an XGBoost model, predicted demand at each station is calculated. These predictions are compared within clusters to identify underperforming and high-demand stations.
#'
#' **4. Bike Reallocation Strategy:**
#' 
#' Stations with excess or insufficient demand are identified, and a reallocation strategy is proposed by calculating the number of bikes to move or receive, ensuring a balance in bike availability.
#'
#' **5. Decision:**
#' 
#' The project delivers a framework for real-time demand prediction and station optimization, empowering CitiBike operators to enhance user experience through efficient resource management.
#'
library(geosphere)
library(ggplot2)
library(caret)
library(xgboost)
library(lubridate)
library(sf)
library(scales)
library(stringr)
library(dplyr)


#'## 1-2. Data Exploration and Cleansing
#'
# Load the sample Citibike data for 2023
DF <- read.csv('sample_citibike_2023.csv')

# Calculate the distance between start and end stations using the Haversine formula
DF <- DF %>% mutate(
  distance = distHaversine(
    matrix(c(start_lng, start_lat), ncol = 2),  # Starting coordinates
    matrix(c(end_lng, end_lat), ncol = 2)      # Ending coordinates
  )
) %>%
  mutate(started_at = as.POSIXct(started_at, format = '%m/%d/%Y %H:%M'),
         Hour = hour(started_at)

  )

# Process and summarize the Citibike data to create station-level insights
station_data <- DF %>%
  # Convert the start time to a POSIXct datetime format for easier manipulation
  mutate(started_at = as.POSIXct(started_at, format = '%m/%d/%Y %H:%M')) %>%
  # Extract the day of the week and hour of the day from the start time
  mutate(
    day_of_week = weekdays(started_at), # Day of the week
    hour_of_day = hour(started_at)      # Hour of the day (0–23)
  ) %>%
  # Group the data by the start station ID to compute station-level statistics
  group_by(start_station_id) %>%
  # Summarize data for each station
  summarise(
    total_rides = n(),                            # Total # of rides at start station
    avg_distance = mean(distance, na.rm = TRUE),  # Average dist. of rides starting here
    casual_rides = sum(member_casual == "casual"), # Total rides by casual users
    member_rides = sum(member_casual == "member"), # Total rides by members
    peak_hour = which.max(table(hour_of_day)),    # The hour of day with the most rides
    rides_weekend = sum(day_of_week %in% c("Saturday", "Sunday")), # Rides on weekends
    start_lat = mean(start_lat, na.rm = TRUE),    # Average latitude of start station
    start_lng = mean(start_lng, na.rm = TRUE)     # Average longitude of start station
  )

## Total Stations
DF %>% distinct(start_station_id) %>% nrow()
## Stations by Borough
DF %>% distinct(start_station_id, boro_name) %>% group_by(boro_name) %>% count(boro_name)
# Average ride distance by Borough
DF %>% group_by(boro_name) %>% summarize(distance = mean(distance, na.rm = T))

# Top 5 average distance by station
T5 <- DF %>% group_by(start_station_id) %>% 
  summarize(distance = mean(distance, na.rm = T)) %>% ungroup() %>%
  arrange(desc(distance)) %>% slice(., 1:5)

DF %>% distinct(start_station_id, start_station_name, boro_name) %>% 
  filter(., start_station_id %in% c(T5$start_station_id)) %>%
  left_join(., T5, by = c('start_station_id'))

ggplot(DF, aes(x = Hour))+
  geom_density(fill = "#69b3a2", color="#e9ecef", alpha=0.8)+
  facet_grid(~boro_name)+
  theme_bw()

# All peak mainly around 4:00 to 5:00 PM
# Manhattan has highest peak of all

#' *Data Exploration Findings:*
#' The dataset contains a total of 2,164 unique start stations. These stations are spread across four boroughs, with Brooklyn having the most stations (700), followed by Manhattan (694), Queens (451), and the Bronx (319). When examining the average ride distance by borough, Brooklyn has the longest average ride distance at 1,993 meters, followed by Manhattan at 1,853 meters, Queens at 1,799 meters, and the Bronx with the shortest average at 1,501 meters.
#'
#' Additionally, the top five stations with the longest average ride distances are concentrated in the Bronx and Queens. The station at 46 Rd & 11 St in Queens has the longest average distance at 11,245 meters, followed by W 238 St & Tibbett Ave in the Bronx with 6,046 meters, and E 188 St & Hughes Ave in the Bronx at 5,702 meters. Other notable stations include 3 Ave & E Tremont Ave in the Bronx (5,872 meters) and 56 Ave & Junction Blvd in Queens (5,053 meters). These stations stand out with significantly higher average ride distances compared to other stations in the dataset.
#'
#'## 3. Training the Model
#'
# Set a random seed for reproducibility
set.seed(123)

# Split the data into training (80%) and testing (20%) sets using random sampling
train_idx <- sample(1:nrow(station_data), size = 0.8 * nrow(station_data))
train_data <- station_data[train_idx, ] %>% ungroup()
test_data <- station_data[-train_idx, ] %>% ungroup()

# Prepare the data for the model by converting the data frames into matrices
train_matrix <- as.matrix(train_data %>% select(-start_station_id, -total_rides,
                                                -start_lat, -start_lng))
test_matrix <- as.matrix(test_data %>% select(-start_station_id, -total_rides,
                                              -start_lat, -start_lng))

# Convert the training and testing matrices into the format required by XGBoost
dtrain <- xgb.DMatrix(data = train_matrix, label = train_data$total_rides)
dtest <- xgb.DMatrix(data = test_matrix, label = test_data$total_rides)

# Train the model
xgb_model <- xgboost(data = dtrain, max_depth = 6, eta = 0.3, nrounds = 100,
                     objective = "reg:squarederror", verbose = 0)

# Makes the predictions from the testing matrix
y_pred <- predict(xgb_model, dtest)

# Calculates the RMSE
rmse <- sqrt(mean((y_pred - test_data$total_rides)^2))
print(paste("RMSE:", rmse))

# Calculate the mean of the target variable (total rides)
baseline_mean <- mean(station_data$total_rides, na.rm = TRUE)

# Calculate RMSE for the baseline model
baseline_rmse <- sqrt(mean((station_data$total_rides - baseline_mean)^2,
                           na.rm = TRUE))
print(paste("Baseline RMSE:", round(baseline_rmse, 2)))


ggplot(data.frame(actual = test_data$total_rides, predicted = y_pred),
       aes(x = actual, y = predicted)) +
  geom_point() +
  geom_abline(color = "red") +
  labs(title = "Actual vs Predicted Popularity", x = "Actual",
       y = "Predicted")

# Residual calculation
residuals <- test_data$total_rides - y_pred

# Residual plot
ggplot(data = test_data, aes(x = y_pred, y = residuals)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Residuals vs. Predicted Values",
       x = "Predicted Values",
       y = "Residuals") +
  theme_minimal()

qqnorm(residuals, main = "Normal Probability Plot of Residuals")
qqline(residuals, col = "red", lwd = 2)

# Residuals vs Fitted Values and NPP display some assumptions may be violated
# trying to remove the outliers to see if this fixes the problem

# Calculate the mean and standard deviation of the average distance
mean_distance <- mean(station_data$avg_distance)
sd_distance <- sd(station_data$avg_distance)

# Calculate Z-scores to identify outliers
station_data$z_score <- (station_data$avg_distance - mean_distance) / sd_distance

# Filter the data to exclude outliers
clean_data <- station_data %>%
  filter(abs(z_score) <= 3)

# Perform K-means clustering on cleaned data based on latitude and longitude
kmeans_result <- kmeans(
  clean_data %>% select(start_lat, start_lng), # Features for clustering
  centers = 5                                 # Number of clusters
)

# Assign cluster labels to the data
clean_data$cluster <- kmeans_result$cluster

# Split the data into training and testing sets (80-20 split)
train_idx <- sample(1:nrow(clean_data), size = 0.8 * nrow(clean_data)) 
train_data <- clean_data[train_idx, ] %>% ungroup() # Training data
test_data <- clean_data[-train_idx, ] %>% ungroup() # Testing data

# Prepare training and testing matrices, excluding non-relevant columns
train_matrix <- as.matrix(train_data %>% 
  select(-start_station_id, -total_rides, -start_lat, -start_lng, -z_score))
test_matrix <- as.matrix(test_data %>% 
  select(-start_station_id, -total_rides, -start_lat, -start_lng, -z_score))

# Convert training and testing data into DMatrix format for XGBoost
dtrain <- xgb.DMatrix(data = train_matrix, label = train_data$total_rides)
dtest <- xgb.DMatrix(data = test_matrix, label = test_data$total_rides)

# Train an XGBoost regression model
xgb_model_2 <- xgboost(
  data = dtrain,
  max_depth = 6,          
  eta = 0.3,              
  nrounds = 100,          
  objective = "reg:squarederror",
  verbose = 0
)

# Generate predictions for the testing set
y_pred <- predict(xgb_model_2, dtest)

# Calculate the Root Mean Square Error for the model
rmse <- sqrt(mean((y_pred - test_data$total_rides)^2))
print(paste("RMSE:", rmse))

# Calculate the baseline RMSE using the mean as a simple prediction
baseline_mean <- mean(station_data$total_rides, na.rm = TRUE)
baseline_rmse <- sqrt(mean((station_data$total_rides - baseline_mean)^2,
                           na.rm = TRUE))
print(paste("Baseline RMSE:", round(baseline_rmse, 2)))

# Scatter plot of actual vs. predicted values
ggplot(data.frame(actual = test_data$total_rides, predicted = y_pred), 
       aes(x = actual, y = predicted)) +
  geom_point() +
  geom_abline(color = "red") +
  labs(
    title = "Actual vs Predicted Popularity",
    x = "Actual Total Rides",
    y = "Predicted Total Rides"
  )

# Calculate residuals (differences between actual and predicted values)
residuals <- test_data$total_rides - y_pred

# Plot residuals vs. predicted values
ggplot(data = test_data, aes(x = y_pred, y = residuals)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Residuals vs. Predicted Values",
    x = "Predicted Values",
    y = "Residuals"
  ) +
  theme_minimal()

# Normal probability plot of residuals
qqnorm(residuals, main = "Normal Probability Plot of Residuals")
qqline(residuals, col = "red", lwd = 2)


##--New Data for Predicting 2024 Demand
DF2 <- read.csv('CitiBike_NewData.csv')

# Calculate distance between start and end stations
DF2 <- DF2 %>% 
  mutate(distance = distHaversine(matrix(c(start_lng, start_lat), ncol = 2),
                                  matrix(c(end_lng, end_lat), ncol = 2)))

# Read borough boundaries
borough_geojson <- st_read("Borough Boundaries.geojson", quiet = TRUE)

# Assign borough to stations
stations_boros <- DF2 %>%
  group_by(start_station_id) %>%
  summarise(longitude = mean(start_lng, na.rm = TRUE),
            latitude = mean(start_lat, na.rm = TRUE))

stations_sf <- st_as_sf(stations_boros, coords = c("longitude", "latitude"),
                        crs = 4326)
stations_with_borough <- st_join(stations_sf, borough_geojson) %>%
  select(start_station_id, boro_name)

DF2 <- DF2 %>% 
  left_join(stations_with_borough, by = 'start_station_id')

station_data <- DF2 %>%
  mutate(started_at = as.POSIXct(started_at, format = '%Y-%M-%d %H:%M:%OS')) %>%
  mutate(day_of_week = weekdays(started_at), hour_of_day = hour(started_at)) %>%
  group_by(start_station_id, start_station_name, boro_name) %>%
  summarise(
    total_rides = n(),
    avg_distance = mean(distance, na.rm = TRUE),
    casual_rides = sum(member_casual == "casual"),
    member_rides = sum(member_casual == "member"),
    peak_hour = which.max(table(hour_of_day)),
    rides_weekend = sum(day_of_week %in% c("Saturday", "Sunday")),
    start_lat = mean(start_lat, na.rm = TRUE),
    start_lng = mean(start_lng, na.rm = TRUE)
  ) %>% ungroup()

#'## 4. Bike Reallocation
#'
# Perform K-means clustering
kmeans_result <- kmeans(station_data %>% select(start_lat, start_lng), 
                        centers = 5)
station_data$cluster <- kmeans_result$cluster

# Prepare data matrix for prediction
new_matrix <- as.matrix(station_data %>% select(-start_station_id, -total_rides,
                                                -start_lat, -start_lng,
                                                -start_station_name,
                                                -boro_name))
dtest_new <- xgb.DMatrix(data = new_matrix, label = station_data$total_rides)

# Predict demand using pre-trained model
predicted_demand <- predict(xgb_model_2, dtest_new)
station_data$predicted_demand <- predicted_demand

##-Post Prediction Analysis

# Visualize demand by cluster
ggplot(station_data, aes(x = cluster, y = predicted_demand,
                         color = as.factor(cluster))) +
  geom_point() +
  labs(title = "Predicted Demand by Cluster", x = "Cluster", y = "Predicted Demand")

# Calculate average demand by cluster
cluster_avg_demand <- station_data %>%
  group_by(cluster) %>%
  summarise(cluster_avg = mean(predicted_demand))

# Identify underperforming and high-demand stations
underperforming_stations <- station_data %>%
  left_join(cluster_avg_demand, by = 'cluster') %>%
  filter(predicted_demand < cluster_avg) %>%
  mutate(bike_to_move = cluster_avg - predicted_demand)

# Underperforming stations by borough
underperforming_stations %>% group_by(boro_name) %>% 
  count(boro_name)

# Top 5 underperforming stations
underperforming_stations %>% arrange(predicted_demand) %>% 
  slice(1:5) %>% select(start_station_name,
                        predicted_demand)

high_demand_stations <- station_data %>%
  left_join(cluster_avg_demand, by = 'cluster') %>%
  filter(predicted_demand > cluster_avg) %>%
  mutate(bike_to_receive = predicted_demand - cluster_avg)

# High demand stations by borough
high_demand_stations %>% group_by(boro_name) %>% 
  count(boro_name)

# Top 5 high-demand stations
top_5_predicted_stations <- high_demand_stations %>%
  arrange(desc(predicted_demand)) %>%
  slice(1:5)

top_5_predicted_stations %>% select(start_station_name, boro_name,
                                    predicted_demand)

high_demand_locations <- DF2 %>%
  filter(., start_station_id %in% top_5_predicted_stations$start_station_id & 
           start_lat %in% c(top_5_predicted_stations$start_lat)) %>%
  distinct(start_station_id, start_lng, start_lat)

stations_sf <- st_as_sf(high_demand_locations, 
                        coords = c("start_lng", "start_lat"), crs = 4326)
stations_sf$density <- sapply(stations_sf$geometry, function(point) {
  sum(st_distance(st_sfc(point, crs = st_crs(stations_sf)), 
                  stations_sf$geometry) < units::set_units(500, "meters"))
})

# Plot top 5 stations on map
ggplot() +
  geom_sf(data = borough_geojson, fill = 'white', color = "black") +
  geom_sf(data = stations_sf, aes(size = density), color = "blue", alpha = 0.4) +
  scale_size_continuous(range = c(1, 10)) +
  theme_minimal() +
  labs(title = "Top 5 High-Demand Station Locations")+
  theme(legend.position = 'none')

# Cost Analysis
costs <- data.frame(member_casual = c('casual','casual', 'member', 'member'),
         rideable_type = c('electric_bike', 'classic_bike','electric_bike',
                           'classic_bike'),
         bike_unlocks = c(4.79, 4.79,0,0),
         cost_per_min = c(.36, 0, .24, 0))

DF2 <- DF2 %>% left_join(., costs, by = c('member_casual', 'rideable_type')) %>% 
  mutate(ended_at = as.POSIXct(ended_at, format = '%Y-%m-%d %H:%M:%OS')) %>% 
  mutate(total_time = as.numeric(difftime(ended_at, started_at, units = "mins"))) %>% 
  mutate(cost = bike_unlocks+(cost_per_min*total_time))

predicted_revenue <- DF2 %>% group_by(start_station_id) %>% 
  summarize(avg_cost = mean(cost, na.rm = T)) %>% ungroup() %>% 
  left_join(station_data, ., by = c('start_station_id')) %>% 
  mutate(predicted_revenue = predicted_demand*avg_cost)

predicted_revenue %>% arrange(desc(predicted_revenue)) %>%
  filter(., start_station_name != '') %>% 
  slice(1:5) %>% 
  mutate(start_station_name = str_replace(start_station_name, 'Central', 'Cent.')) %>% 
  mutate(station_label = paste0(start_station_name, '\n', '(', boro_name,')')) %>% 
  mutate(predicted_revenue_monthly = predicted_revenue/12) %>% 
  ggplot()+
  geom_bar(aes(x = station_label, y = predicted_revenue_monthly), 
           stat = 'identity', fill = 'skyblue', color = 'black')+
  scale_y_continuous(breaks = pretty_breaks(), 
                     labels = label_dollar(prefix = "$", accuracy = 1))+
  theme_bw()+
  theme(axis.title = element_blank(),
        axis.text.y = element_text(size = 8))+
  coord_flip()+
  ggtitle('Top 5 Project Monthly Revenue Stations')

predicted_revenue %>% arrange(desc(predicted_revenue)) %>% 
  filter(., start_station_name != '') %>% 
  group_by(boro_name) %>% 
  slice(1:5) %>% 
  mutate(predicted_revenue_monthly = predicted_revenue/12) %>% 
  mutate(start_station_name = str_wrap(start_station_name, width = 15)) %>% 
  ggplot()+
  geom_bar(aes(x = start_station_name, 
               y = predicted_revenue_monthly, fill = boro_name),
           stat = 'identity', color = 'black')+
  scale_y_continuous(breaks = pretty_breaks(), 
                     labels = label_dollar(prefix = "$", accuracy = 1))+
  facet_wrap(~boro_name, ncol = 2, scales = "free")+
  theme_bw()+
  theme(axis.title = element_blank(), legend.position = 'none',
        axis.text = element_text(size = 7),
        plot.margin = margin(0,0,0,0, "cm"))+
  coord_flip()+
  ggtitle('Top 5 Project Monthly Revenue Stations By Borough')


predicted_revenue %>% arrange(desc(predicted_revenue)) %>% 
  filter(., start_station_name != '') %>% 
  group_by(boro_name) %>% 
  slice(1:5) %>% 
  mutate(predicted_revenue_monthly = predicted_revenue/12,
         predicted_revenue_monthly = label_dollar()(predicted_revenue_monthly)) %>% 
  select(Station = start_station_name, Borough = boro_name,
         Monthly_Revenue_Dollars = predicted_revenue_monthly)

predicted_revenue %>% arrange(desc(predicted_revenue)) %>% 
  filter(., start_station_name != '') %>% 
  group_by(boro_name) %>% 
  summarize(predicted_revenue_monthly = mean(predicted_revenue/12, na.rm = T)) %>% 
  ungroup() %>% 
  ggplot()+
  geom_bar(aes(x = boro_name, y = predicted_revenue_monthly),
           stat = 'identity', fill = 'skyblue', color = 'black')+
  scale_y_continuous(breaks = pretty_breaks(), 
                     labels = label_dollar(prefix = "$", accuracy = 1))+
  theme_bw()+
  theme(axis.title = element_blank(), legend.position = 'none')+
  ggtitle('AVG Monthly Revenue Borough')

# Highest Predicted Avg Monthly Revenue Station
DF2 %>% filter(., start_station_name == 'Central Park S & 6 Ave') %>% 
  group_by(member_casual, rideable_type) %>% count(rideable_type)

# AVG Manhattan Station in October 2024
DF2 %>% filter(., boro_name == 'Manhattan') %>% 
  group_by(start_station_id, member_casual, rideable_type) %>% 
  count(rideable_type) %>% ungroup(start_station_id) %>% 
  summarize(AVG = mean(n,na.rm = T))
