# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 4 - West Nile Virus Prediction

## Group member

Larb|B.B.|PuNt|
|---|---|---|
![](image/Kamlarp.png)|![](image/Chonnathai.png)|![](image/Ponparis.png)

## Introduction

This is project 4 of of the General Assembly Data Science Immersive course, Project West Nile Virus Prediction. The objective of this project is to demostrate the data science process to provide the prediction of category data, if the virus is found or not. 


## Problem statement

West Nile Virus is a deadly virus transmitted by mosquitoes. Once it infects humans, 20% of individuals may develop symptoms, ranging from a persistent fever to severe neurological illnesses that can lead to death. The City of Chicago and the Chicago Department of Public Health (CDPH) have joined forces to control the spread of mosquitoes and, consequently, the spread of the West Nile Virus.

Our team, consisting of amateur data scientists, has entered a competition hosted by the City of Chicago. Our goal is to develop a predictive model for the occurrence of the virus. This model will assist the City of Chicago in planning pesticide spraying operations and conducting cost-benefit analyses to justify their spraying plans. The results will be presented to members of the Centers for Disease Control and Prevention (CDC), including biostatisticians and epidemiologists.

## The Data

4 set of data were given for model development.
1. Train data
2. Test data 
3. Weather data
4. Pesticides sprayed data

The data dictionary for each data are as follow:

Train data
Column| Description|
|---|---|
Id| the id of the record
Date| date that the WNV test is performed
Address| approximate address of the location of trap. This is used to send to the GeoCoder. 
Species| the species of mosquitos
Block| block number of address
Street| street name
Trap| Id of the trap
AddressNumberAndStreet| approximate address returned from GeoCoder
Latitude, Longitude| Latitude and Longitude returned from GeoCoder
AddressAccuracy| accuracy returned from GeoCoder
NumMosquitos| number of mosquitoes caught in this trap
WnvPresent| whether West Nile Virus was present in these mosquitos. 1 means WNV is present, and 0 means not present. 

Test data
Column| Description|
|---|---|
Id| the id of the record
Date| date that the WNV test is performed
Address| approximate address of the location of trap. This is used to send to the GeoCoder. 
Species| the species of mosquitos
Block| block number of address
Street| street name
Trap| Id of the trap
AddressNumberAndStreet| approximate address returned from GeoCoder
Latitude, Longitude| Latitude and Longitude returned from GeoCoder
AddressAccuracy| accuracy returned from GeoCoder

Weather data
Column| Description|
|---|---|
Station| Weather station ID
 Date| Date
 Tmax| Maximum Temperature
 Tmin| Minimum Temperature 
 Tavg| Average Temperature
 Depart| Departure from normal
 DewPoint| dew point temperature
 WetBulb| Wetbulb temperature
 Heat| Heating start in July
 Cool| Colling start in Jan
 Sunrise| time of sunrise
 Sunset| time of sunset
 CodeSum| Code for Tornado/Waterpout
 Depth| Depth of snow (in inch)
 Water1| Depth of water
 SnowFall| T = Trace, M = Missing
 PrecipTotal| Rainfall and melt snow
 StnPressure| Average pressure
 SeaLevel| Sea level pressure
 ResultSpeed| Wind speed (miles per hour)
 ResultDir| resultant direaction
 AvgSpeed| Average wind speed

 Spray data
 Column| Description|
|---|---|
Date| Date of spray
Time| Time of spray
Latitude| Latitude where spray
Longitude| Longitude where spray


## Exploratory Data Analysis and Data pre-processing

##### Check and clean the data
- Only feature "time" in spray data contains "null" value. Since there is no time record in other dataset. So, consider drop this feature.
- The spray record contain data in year 2011, 2013. But test data has no record in those 2 years. So consider not using spray data for now.
- We check the duplicated data in all dataset, and found duplicated data in Train data. So drop the duplicate from dataset.
- Weather data has no "null" value, but virtually, we see blank and missing data; so we consider replacing them with "null".
![](image/weather_missing.png)
- There are total less than 400 record with replaced null value, out of total 2944 record in weather. This is about 20%, so we consider dropping them.
- We, then, convert all data in weather data set to integer or float.
- We also drop 'Codesum' column, on the assumption that this is tornado warning, this not happen very frequent in Chicago.
- There are 2 stations; Since both station class has about the same record and we don't know the location of each station. we split both and re-merge them together with date increase the variance for better prediction when develop the model.
- We, then, merge weather with train and test data.
- we, then convert and split datetime to year, month, day for both data set.
- We, finally, have 2 data set. one is train data and another one is test data (to be used for prediction submission).

##### EDA the data
- Let's check if there is any correlation between data

![](image/heatmap.png)
![](image/heatmap2.png)

- High correlation occur among weather data; this high correlation show the sign of multicollinearity.
- The month data, temperature related data, location related such as lat/long has notable correlation with WnvPresent.

- Check the present of virus across each month of year and each year of record
![](image/present_month.png)
![](image/present_year.png)

It is clear that the virus is likely to be found during summer. It is likely that it might related to temperature. So let's check the presence of virus against temperature.
![](image/temp.png)

This is align with previous finding. The virus is likely to be found in the higher temperature.

- Next, let's check if there is any significant of the location across Chicago.
![](image/map.png)

- Some areas may have more presence than other, but it look evenly spread across city.

- Eventhough Spray data doesn't look useful, but let's check them for certian. Check if there is any impact from spray
![](image/spray_month.png)

The spray was done in 2011 and 2013. The graph shows that in 2013, the virus present is the highest. The spray not working or did they spray in the wrong place.

![](image/distance.png)

We do the calculation to find the distance (in kilometer) from the spray lat/long to the closest trap. The result in the above graph shows that the spray location is further away range from 1 - 40 km away. So this might be a reason why the spray didn't work.

##### EDA Summary
- For train data, the baseline score is 5% Present (virus found) and 95% Non-present (virus not found); therefore train data is very imbalance; we will do the oversampling with SMOTE technique later in the modeling.
- For spray data, we  will not use it because it is not reliable.
- For weather data, we will merge it with train data because the weather data, especially temperature, appear to be strong predictors.

##### As a result of EDA, Data preprocessed for model development
- For features that contain text such as spicies and traps, we will use label code to convert them to be numerical features.
- In conclustion, we have 36 X variables (features) and 1 Y target variable to work for model development.
- 36 X variables includes 
       'Species', 'Block', 'Street', 'Trap', 'Latitude', 'Longitude',
       'AddressAccuracy', 'WnvPresent', 'Tmax_x', 'Tmin_x', 'Tavg_x',
       'DewPoint_x', 'WetBulb_x', 'Heat_x', 'Cool_x', 'PrecipTotal_x',
       'StnPressure_x', 'SeaLevel_x', 'ResultSpeed_x', 'ResultDir_x',
       'AvgSpeed_x', 'Tmax_y', 'Tmin_y', 'Tavg_y', 'DewPoint_y', 'WetBulb_y',
       'Heat_y', 'Cool_y', 'PrecipTotal_y', 'StnPressure_y', 'SeaLevel_y',
       'ResultSpeed_y', 'ResultDir_y', 'AvgSpeed_y', 'day', 'month', 'year'
- 1 Y variable is 'WnvPresent'

## Modeling and result

##### Selecting best model algorithm
This is classification case, predicting the probability of label. So our group start with using 7 classification models to select the best model, and to use this best model for futher tuning
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. AdaBoost
5. Bagged Decision Tree
6. Gradient Boosting
7. SVM
8. XGBoost

Here are result

![](image/result1.png)   ![](image/result2.png)

We use AUC-ROC score for evaluating th best model. XGBoost got the best score. So we decide to pick XGBoost for further tuning using gridsearch.

##### Selecting SMOTE vs Non-SMOTE
Since the train data is very imbalance, so we decide to test between
    1) model using original data (Non-SMOTE)
    2) model with resampling data (SMOTE).

The SMOTE one has a better result while train the model.

##### Feature Importance Analysis to explore futher
To find out what kind of prediction our model has done, we check the confusion matrix and the decision tree.

 ![](image/confuse.png)

 ![](image/tree1.png)

Based on confusion matrix, Sensitivity (TP/TP+FN) is low, only 5%. The result indicate that the True Positive over Total actual positive case is very poor. 
 
Based on the decision tree, we see that month, latitude, and temperature have strong impact, and we decide to do further feature engineering  on temperature.

As a result we created 34 more features by creating the bin for each 8 Farenhiet degree of Tmax, Tmin, Tavg, and Wetbulb as features, for example, 'wetbulb_x_bin_bin8'

##### Final model tuning
With additionl features which apply XG Boost, the results improved. Again the non-SMOTE actually got better score than SMOTE.

 ![](image/finalscore.png)


#### the model challenges and limitation
1) the pesticide spraying is not effective because it was done faraway from Trap especially, so the spray data is not usable.
2) the train data is highly imbalance; the imbalance impact the prediction result.
3) the train data is not continuous from year to year; we should have continous yearly data instead.
4) the test data lack WnvPresent data; and the lack of this data make us unable to generate spatial feature such as
       - Distance between trap and WnvPresnt
       - Count of WnvPresent in xx KM during last xx days


## Conclusion & Recommendation

### Predictors
In conclusion, month, latitude, and temperature, are the strongest predictors for virus presence. 

### Wnv control plan based on our model prediction
The city should encourages practicing the three “R’s” – reduce, repel, and report based on our model prediction.

REPORT: 
we can use our prediction to check predicted locations where you see water sitting stagnant for more than a week such as roadside ditches, flooded yards, and similar locations that may produce mosquitoes.  

REDUCE: 
we can use our prediction to make sure that in the predicted areas ...

            - sprays are used preventively 
            - doors and windows have tight-fitting screens, and be repaired or replaced for those that have tears or other openings 
            - doors and windows are shut
            - all sources of standing water where mosquitoes can breed

REPEL: 
we can use our prediction to make sure that in the residents in the predicted areas wear shoes and socks, long pants and a light-colored, long-sleeved shirt, and apply an EPA-registered insect repellent that contains DEET, picaridin, oil of lemon eucalyptus, IR 3535, para-menthane-diol (PMD), or 2-undecanone according to label instructions.  Consult a physician before using repellents on infants.

Source: https://dph.illinois.gov/resource-center/news/2023/june/west-nile-virus-reported-in-four-illinois-counties-so-far-in-202.html

### Cost and benefit analysis of Wnv control plan based on our model prediction

#### Benefits:
The benefits of using machine learning to control West Nile virus (WNV) control for the government of Chicago, Illinois are numerous and far-reaching. By implementing machine learning, we can be more effectively control WNV, and the city can significantly reduce the risk of WNV transmission, protect the health of its residents, and save money in the long run.

##### 1) Improve the Public Health: 
- Based on statistics in 2022, there were 34 human cases (which are significantly under-reported) and 8 deaths attributed to the disease in the state in 2022, the most in any year since 2018, when there were 17 deaths.
- Using machine learning to predict where and when can reduce the number of WNV cases, and the city can prevent serious illness, hospitalization, and even death. This has a direct impact on the quality of life of Chicago residents and can help to reduce the overall burden on the healthcare system.

Source: https://dph.illinois.gov/resource-center/news/2023/june/west-nile-virus-reported-in-four-illinois-counties-so-far-in-202.html

##### 2) Reduce the indirect Economic Costs: 
- Using machine learning to predict where and when also can control WNV outbreaks which have a significant economic impact on a city. In addition to the direct costs of medical care, WNV can also lead to lost productivity, decreased tourism, and increased anxiety among residents. By preventing WNV outbreaks, the city can save money and help to keep its economy strong.

#### Possible Cost : 
   - purchasing and applying larvicide in the predicted areas: 
             149,533.65 USD
    - working with local municipal governments and local news media for WNV prevention and education: 
            0 USDs (remark: we will do it anyway)
    - investigating mosquito production sites and nuisance mosquito complaints on the predicted site: 
            0 USDs (remark: use local resources) 
    - collecting mosquitoes for West Nile virus testing and also collect sick or dead birds for West Nile virus testing:
            0 USDs (remark: use local resources)

####  Cost Calculation: 
Total size of Chicago area is 28,120 km square
Total size of Wnv presence in Chicago area is 1,665 km square based on using predictive modeling on test data (Remark: assuming that 1 trap which we track WnvPresent cover 1 km square)
Cost of mosquito repellant per 1 km square is 89.81 USD (Remark: assuming that 12 bottles of "Repel 33801-1 Sportsmen Max Insect 6.5-oz Aerosol 40% DEET" cover 1 km square area) 
Cost of mosquito repellant for predicted area (1,665 km square) = 149,533.65 USD 
Cost of total Chicago area (28,120 km square) = 2,302,747 USD
We can save cost of applying larvicide by 2,302,747 USD - 149,533.65 USD = 2,153,213.35 USD

Source:
https://en.wikipedia.org/wiki/Chicago_metropolitan_area
https://www.amazon.com/Repel-33801-Sportsmen-Repellent-40-Percent/dp/B008H5B9UK/ref=sr_1_2?crid=26F23KCJIK9J4&keywords=mosquito%2Brepellent%2Bspray%2Blarge%2Bsize&qid=1701148171&sprefix=mosquito%2Brepellent%2Bspray%2Blarge%2Bsiz%2Caps%2C347&sr=8-2&th=1






