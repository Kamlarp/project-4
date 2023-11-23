# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 4 - West Nile Virus Prediction

## Group member

Larb|B.B.|PuNt|
|---|---|---|
![](image/Kamlarp.png)|![](image/Chonnathai.png)|![](image/Ponparis.png)

## Introduction

This is project 4 of of the General Assembly Data Science Immersive course, Project West Nile Virus Prediction. The objective of this project is to demostrate the data science process to provide the prediction of category data, if the virus is found or not. 


## Problem statement

West Nile Virus is a deadly virus found in mosquitos. One it is infected to human, 20% of people develop symtoms ranging from a persistent fever, to serious neurological illnesses that can result in death. City of chicago and CDPH together want to control the spread of mosquitos, hence control the spread of West Nile Virus.

Our team, as the amature data scientist group, we enter the competition hosted by City of Chicago to develop the model to predict the occurence of virus, so the City of Chicago can use them when they want to plan pesticides spraying as well as the cost-benefit analysis to justify the spraying plan. The result will be presented to members of CDC, including biostatistician and epidemiologists.

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
##### Check data correction and readiness
- Only feature "time" in spray data contains "null" value. Since there is no time record in other dataset. So, consider drop this feature
- The spray record contain data in year 2011, 2013. But test data has no record in those 2 years. So cosider not using spray data for now.
- Check duplicate data in all dataset, and found duplicated data in Train data. So drop the duplicate from dataset
- Weather data has no "null" value, but vitually, we see blank and missing data. Replace them with "null" before consider dropping them
![](image/weather_missing.png)
- There are total less than 400 record with replaced null value, out of total 2944 record in weather. This is about 20%, so cosider to drop them.
- convert all data in weather to integer or float
- drop Codesum column, on the assumption that this is tornado warning, this not happen very frequent in Chicago.
- There are 2 station, split them form another set of features. This is to expand the features in order to increase the variance for better prediction when develop the model
- Merge weather with train and test data
- convert and split datetime to year, month, day for both data set
- We are not have 2 data set intead of 4. 1 of 2 is train data and another is test data (to be used for prediction submission)

##### EDA
- Let's check if there is any correlation between data

![](image/heatmap.png)
![](image/heatmap2.png)

There are no significant correlation between WnvPresent, Nummosquitos and other datas. The correlation only occur amoung weather data. Which is common.

- Check the present of virus across each month of year and each year of record
![](image/present_month.png)
![](image/present_year.png)

This is clear than the virus is likely to be found during summer. This is likely that it might related to temperature. So let check the present of virus against Temperature
![](image/temp.png)

This is align with previous finding. The virus is likely to be found in the higher temperature

- Next, let's check if there is any significant of the location across Chicago
![](image/map.png)

Some area may have more present that other, but it look evenly spread across city.

- Eventhought Spray data doesn't look useful, but let's check them for certian. Check if there is any impact from spray
![](image/spray_month.png)

The spray was done in 2011 and 2013. The graph shows that in 2013, the virus present is the highest. The spray not working or did they spray in the wrong place.

![](image/distance.png)

We do the calculation to find the distance (in kilometer) from the spray lat/long to the closest trap. The result in the above graph shows that the spray location is further away range from 1 - 40 km away. So this might be a reason why the spray didn't work.



##### Summary of EDA
- The correlation between features and WnvPresent are minimal. Most of correlation found are amoung each weather features.
- It is clear that Virus is likely to be found during Jun - Oct. The highest is Aug. 
- This is also matched the information of temperature where Virus is likely to be found
- It is unclear to relate geolocation such Block to the present of virus.
- In train data, the baseline score is 5% WnvPresent and 95% not present. This is very unbalance train data

The only strong indicators are weather, so let's select features heavily rely on weather features

##### Data preparation for model development
- Address and street address are duplicate to the lat/long and even block/trap, so consider to drop them
- Some feature with text present such spicies and traps are not number, so use hotcoded to convert them to number so the model can work with these features
- now we have 36 features (and 1 classification) to work with Model development


## Modeling and result
##### Modeling
This is classification case, predicting the probability of label. So our group start with using 8 classification model as follow. Aiming to get the best result score to use them for futher optimization
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. AdaBoost
5. Gradient Boosting
6. Bagged Decision Tree
7. SVM
8. XGBoost

Here are result

![](image/result1.png)   ![](image/result2.png)

We're using AUC-ROC score for evalulation. XGBoost got the best score. So we decide to pick XGBoost as the model for further hyperparameter optimization using gridsearch. After the optimization, let's check the result

##### Error Analysis
Since the train data is very unbalance, so we decide to run twice, one with prepared data and another with resampling data using SMOTE method.
The SMOTE has a better resul. And we use both models to predict and submit to kaggle. Surprisingly non-SMOTE gave a better score on Kaggle at 0.7068 and SMOTE model is 0.6778. 

To find out what kind of prediction our model has done, we use the splited test data to check the confussion matrix and the tree
 ![](image/confuse.png)

  ![](image/tree1.png)

 Sensitivity (TP/TP+FN) is not good at all, only 5%. Due to baseline, model is likely to predict mostly no WMV. The AUC-ROC is also at 0.53. Both of them indicate the True Positive over Total actual positive case is very poor. 
 
 Then let's check the tree. We can see that this is align with our assumption earlier that weather might have correlation, because the tree shows in the early branch of tree.

 
##### Feature optimization
From the error analysis we found that there could be a correlation between the temperature and present of virus. so let got back and check on those feature, and perform features engineering aiming to improvde the model performance. 

We ending up created 34 more features from creating the bin for every 8 degree F of Tmax, Tmin, Tavg, and Wetbulb. 

Let's apply selected model to these data.

##### Model optimization

After using new features data, the result of the model is improved. AUC-ROC of both splited train and test data of SMOTE went to 0.6. and When we made prediction and submitted to Kaggle, the result also improved. Again the non-SMOTE actually got better score than SMOTE

 ![](image/finalscore.png)


## Conclusion
From our result, we found that the factor that has impact on the present of the virus is likely to be weather, or to be more presice the temperature. When we focus our feature engineer on weather factor, we got a better result. We would recommend that if the City is to plan pesticide spraying, they should concentrate to do it during summer and perhaps change the pesticide, because from spray data, it doesn't look very effective.

#### Limitation
We concern about the impact of pesticide spray that it not effective. But we're not sure yet, because it might be that spray was done further from Trap, so we didn't detect any effect of spray. Perhaps more data on spray would prove otherwise. Also the train data was strongly imbalance, where the test baseline shows very well balance case. 

#### Cost and Benefit analysis
##### WNV control with machine learning  

The Illinois Department of Public Health (IDPH) encourages the public to Fight the Bite by practicing the three “R’s” – reduce, repel, and report:

REPORT: we can use our prediction to check predictted locations where you see water sitting stagnant for more than a week such as roadside ditches, flooded yards, and similar locations that may produce mosquitoes.  

REDUCE: we can use our prediction to make sure that in the predicted areas ...
            - sprays are used preventively 
            - doors and windows have tight-fitting screens, and be repaired or replaced for those that have tears or other openings 
            - doors and windows are shut
            - all sources of standing water where mosquitoes can breed, including water in bird baths, ponds, flowerpots, wading pools, old tires, and any other containers are taken care of

REPEL: we can use our prediction to make sure that in the resident in the predicted areas wear shoes and socks, long pants and a light-colored, long-sleeved shirt, and apply an EPA-registered insect repellent that contains DEET, picaridin, oil of lemon eucalyptus, IR 3535, para-menthane-diol (PMD), or 2-undecanone according to label instructions.  Consult a physician before using repellents on infants.

##### Benefits:

The benefits of using machine learning to control West Nile virus (WNV) control for the government of Chicago, Illinois are numerous and far-reaching. By implementing machine learning, we can be more effectively control WNV, and the city can significantly reduce the risk of WNV transmission, protect the health of its residents, and save money in the long run.

##### 1) Improve the Public Health: 

- Based on statistics in 2022, there were 34 human cases (which are significantly under-reported) and 8 deaths attributed to the disease in the state in 2022, the most in any year since 2018, when there were 17 deaths.
- Using machine learning to predict where and when can reduce the number of WNV cases, and the city can prevent serious illness, hospitalization, and even death. This has a direct impact on the quality of life of Chicago residents and can help to reduce the overall burden on the healthcare system.

##### 2) Reduce the indirect Economic Costs: 

- Using machine learning to predict where and when also can control WNV outbreaks which have a significant economic impact on a city. In addition to the direct costs of medical care, WNV can also lead to lost productivity, decreased tourism, and increased anxiety among residents. By preventing WNV outbreaks, the city can save money and help to keep its economy strong.

##### Cost: 
 The possible cost includes increasing the following activities to ensure that we include the areas by machine learning
    - purchasing and applying larvicide,
    - working with local municipal governments and local news media for WNV prevention and education, 
    - investigating mosquito production sites and nuisance mosquito complaints. 
    - collecting mosquitoes for West Nile virus testing and also collect sick or dead birds for West Nile virus testing.











