# DS4320 Project 2: Predicting 911 Call Volumes in Austin, TX

This repository contains a machine learning dataset and modeling pipeline for predicting daily emergency 911 call volume in the Baker sector of Austin, Texas (the APD dispatch zone covering the University of Texas campus and stadium). The dataset consists of 17,399 individual call records spanning 2024 and 2025, sourced from the Austin Police Department's public open data portal and enriched with daily weather data from the Open-Meteo API and binary event flags for major local events including UT Longhorns home football games, SXSW, Austin City Limits, UT move-in weekend, spring break, and graduation. All enriched documents are stored in MongoDB Atlas. The pipeline includes two notebooks: a data ingestion notebook that processes raw APD data and loads it into MongoDB, and a pipeline notebook that queries MongoDB, aggregates call records to daily counts, and trains both a linear regression baseline and an XGBoost model to predict daily call volume. 

Eva Winston

vxm2ek

[![DOI](https://zenodo.org/badge/1083840299.svg)](https://doi.org/10.5281/zenodo.19872802)

[Press Release](https://github.com/evamaiwinston/ds4320project2/blob/main/PressRelease.md)

[Solution Pipeline](https://github.com/evamaiwinston/ds4320project2/blob/main/pipeline.ipynb)

[License](https://github.com/evamaiwinston/ds4320project2/blob/main/LICENSE)

## Problem Definition
Initial problem: Allocating emergency response resources

Refined problem: For efficient emergency resource deployment in Austin, TX, can we predict 911 emergency call volume surges in the Baker district using historical call records, and event (football game, music festival, etc.) dates?

My motivation for this project is that I wanted to do something different because I have done a lot of energy projects and have seen a lot of diabetes datasets. I think that allocating emergency response resources is a very helpful use of machine learning and data analysis. I figured that certainly sporting events, concerts, and other large public gatherings — like UT Longhorns home football games, SXSW, and Austin City Limits — influence the amount of emergency 911 calls in Austin, Texas. I was interested in seeing if I could use those features in conjunction with other features like weather and historical 911 call data to predict when the most Priority 0 and Priority 1 calls would occur in the Baker sector, the APD dispatch zone covering UT's campus and stadium. I am motivated to create a dataset to solve this problem because I think it is a worthwhile cause, and it is something that could actually help prevent negative outcomes of emergencies if the features are as predictive as they seem.

The first part of the rationale for the refinement was determining how emergency resources are allocated. 911 is the hub for taking emergency calls and dispatching firefighting units, ambulance units, etc. So, I figured that 911 call volumes would be a good metric to target in order to improve resource allocation. Then, I decided to refine the scope to a specific city. I have done a lot of projects about New York City and I just did one about California. I chose Austin, Texas because it has a rich publicly available 911 dataset through the Austin Open Data Portal with over 3.9 million records. I think Austin is a city I would really enjoy living in if I ever get the chance. It has a vibrant sports and events scene, including UT Longhorns football, SXSW, and Austin City Limits, which also makes it an ideal city to examine whether large public gatherings correlate with emergency call volumes.

[Austin Emergency Services: Predicting 911 Call Surges Before They Happen](https://github.com/evamaiwinston/ds4320project2/blob/main/PressRelease.md)

## Domian Exposition
| Term | Definition |
|---|---|
| 911 Call Volume | How many Priority 0 and Priority 1 calls came in during a given day in the Baker sector |
| Priority 0 | Highest urgency calls — immediate life threat requiring immediate response |
| Priority 1 | Urgent calls requiring a quick but non-emergency response |
| Baker Sector | The APD dispatch zone covering UT Austin's campus, stadium, and surrounding area |
| Sector | A named APD dispatch zone used to divide Austin into operational response areas |
| Response Time | How long it takes from when someone calls 911 to when help actually arrives |
| Event Feature | A binary flag indicating whether a given day coincides with a major event like SXSW, ACL, or a UT home football game |
| Temporal Features | Time-based variables used as model inputs — like day of week, month, and day of year |
| Pre-positioning | Moving emergency units to high-demand areas before a surge hits, instead of reacting after |
| Weather Features | Daily meteorological variables like max temperature, precipitation, and wind speed used to capture weather effects on call volume |

This project is set in the world of public safety and city emergency management. Austin's 911 system handles millions of calls every year covering everything from medical emergencies to traffic incidents to criminal disturbances. The big problem is that emergency resources like ambulances, fire trucks, and police units are usually deployed based on fixed plans or a dispatcher's mental model, not on any kind of quantified prediction. But Austin publishes all of its 911 call data publicly through the Austin Open Data Portal, which creates the opportunity of combining that call history with weather data and local event schedules, like UT Longhorns football games, SXSW, and Austin City Limits, to build a model that predicts when call volume is going to spike in the Baker sector. The goal is to help the city get ahead of surges instead of scrambling to react to them.

[Link to background readings folder](https://myuva-my.sharepoint.com/:f:/g/personal/vxm2ek_virginia_edu/IgBpEyMffAFdS6bSyZuGip5nAR5uytLvtAaEXUQNDRoT0Y4?e=DOcEt5)


| # | Title | Description | Link |
|---|---|---|---|
| 1 | Surveillance and Predictive Policing Through AI — Deloitte | Overview of how AI and predictive tools are being used in public safety, including the ethical and operational tradeoffs cities face when deploying these systems | [Link](https://myuva-my.sharepoint.com/:u:/g/personal/vxm2ek_virginia_edu/IQCx-bDC2Tt6QpGdjavpVVInAQ9f0-x3JCR8sMG2HSuqzm4?e=0374K9) |
| 2 | An Algorithm for EMS Response | UT research on using algorithms to improve EMS resource deployment and response times| [Link](https://myuva-my.sharepoint.com/:u:/g/personal/vxm2ek_virginia_edu/IQB4Trc1mzH3Trg8Nd1C6R38AaDl8i5_7GexsVafp1SmyCc?e=Jzj8RW) |
| 3 | Short-Term Forecasting of Emergency Medical Services Demand — Shahidian et al. | Academic paper exploring machine learning models for predicting EMS call demand in the short term — directly relevant to our prediction approach | [Link](https://myuva-my.sharepoint.com/:u:/g/personal/vxm2ek_virginia_edu/IQAJjEwTfu5kTY70H2CafOVvATtj3vDThQdx89OatxvSoyI?e=xivysA) |
| 4 | Inside Austin's SXSW Event Operations Center: How the city is coordinating safety and traffic | Local news piece showing how Austin deploys a multi-agency emergency operations center specifically for SXSW, confirming the festival's impact on public safety resources | [Link](https://myuva-my.sharepoint.com/:u:/g/personal/vxm2ek_virginia_edu/IQCC_KbS26ulRp9LxehlY3rKAUF3Vt0Q4XSZXLxvYDbYjEo?e=rscg8x) |
| 5 | Forecasting the daily demand for emergency medical ambulances in England and Wales: a benchmark model and external validation | Peer-reviewed study building a benchmark model to predict daily emergency ambulance demand using historical data | [Link](https://myuva-my.sharepoint.com/:u:/g/personal/vxm2ek_virginia_edu/IQDE3b7zK16ET4VaOGRrBheDARce9WXn12yIloCP5X3qD8E?e=Musvpd) |

## Data Creation

The raw data acquisition began with a public CSV file sourced from the Austin Police Department's open data portal, which publishes historical 911 calls for service. The raw dataset contains over 3.9 million records spanning 2017 to 2026. For this project, I filtered records to Priority 0 and Priority 1 calls only a.k.a. the two highest urgency levels,and further narrowed the geographic scope to the Baker sector, the APD dispatch zone covering UT Austin's campus and stadium. This resulted in 17,399 individual call records across 2024 and 2025. Rather than a random sample, I used the full filtered dataset to preserve temporal integrity. Each call record was enriched with daily weather data from the Open-Meteo historical weather API, which is free to access and requires no API key. Weather variables include daily max and min temperature, precipitation, wind speed, and weather code. Additionally, I added binary event features for each record based on its date, flagging occurrences of UT home football games, SXSW, Austin City Limits, UT move-in weekend, spring break, graduation, and federal holidays. All enriched documents were loaded into MongoDB Atlas, with 2025 records designated as the training split and 2024 records as the test split.



| File | Description | Link |
|---|---|---|
| data_ingestion.ipynb | Raw Austin Police Department 911 calls acquisition and ingestion into MongoDB Atlas | [Link to code](https://github.com/evamaiwinston/ds4320project2/blob/main/data_ingestion.ipynb) |

The data acquisition and preparation process involved several judgment calls that are important to document. The decision to filter the dataset to Priority 0 and Priority 1 calls only was made as these represent true emergency dispatches where resource allocation decisions are most consequential. Lower priority calls were excluded because they are less likely to strain emergency resources and introduce noise into the prediction target. The decision to scope the analysis to the Baker sector was made because it is the APD dispatch zone containing UT Austin's campus and stadium, making it the most relevant area for testing whether large public events drive 911 call volume. The full filtered dataset of 17,399 records across 2024 and 2025 was used rather than a random sample, preserving the true temporal distribution of incidents. The reasoning for the train/test split based on year was that the test set would contain a full UT football season, allowing proper evaluation of the game day feature. Open-Meteo was selected as the weather source because it is free, requires no API key, and is fully reproducible. Austin, Texas was selected as the city because the Austin Police Department publicly reports a complete and well-documented 911 calls for service dataset, and because Austin's unique combination of a major university, large annual festivals, and a growing urban population makes it an ideal environment for studying how public events influence emergency call volume.

Several potential sources of bias exist in the data collection process for this project. First, the Austin 911 dataset only includes Priority 0 and Priority 1 calls, meaning lower priority calls, non-emergency incidents, and calls resolved over the phone are excluded. This could underrepresent certain incident types or time periods where lower priority calls are more common. Geographic bias is also present since the analysis is scoped entirely to the Baker sector, so findings may not generalize to other APD dispatch zones with different population densities or activity patterns. The weather data may introduce bias because it is sourced from a single coordinate point representing central Austin, which may not perfectly capture weather variation across the broader city. Additionally, the event features only capture officially scheduled and publicly known events, so informal gatherings, spontaneous large crowds, or unregistered events that drive 911 call volume are not recorded, introducing omission bias. Finally, training on a single year (2025) and testing on a single year (2024) means the model may be sensitive to year-specific patterns.

In order to mitigate bias, several strategies were used. To address the exclusion of lower priority calls, the problem statement is clearly scoped to Priority 0 and Priority 1 dispatched incidents only, avoiding any generalizations about overall 911 call volume. Rather than random sampling, the full filtered dataset of 17,399 records was used to preserve the true distribution of incidents across dates, times, and call categories. The geographic limitation to Baker sector is explicitly documented as a design choice rather than a representative sample of all Austin sectors. This sector was selected intentionally because it contains UT Austin's campus and stadium, making it the most relevant zone for the event-based features in this dataset. The weather data bias is addressed in documentation, noting that the Open-Meteo coordinates represent central Austin and serve as an approximation rather than a precise measurement for the entire sector. The absence of informal events is partially mitigated through temporal features such as day of week, is_weekend, and is_holiday, which serve as proxies for elevated social activity even when no formal event is registered. While these strategies do not eliminate all sources of bias, they reduce their impact and are transparently documented so that future users of the dataset understand its limitations.



## Metadata

An outline of the implicit schema is: 
```
schema = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": [
            "_id",
            "priority_level",
            "response_datetime",
            "final_problem_category",
            "split"
        ],
        "properties": {
            "_id": {
                "bsonType": "int",
                "description": "Unique incident number from APD source data, used as document identifier"
            },
            "incident_type": {
                "bsonType": "string",
                "description": "Whether the incident was dispatched or self-initiated"
            },
            "mental_health_flag": {
                "bsonType": "string",
                "enum": ["Mental Health Incident", "Not Mental Health Incident"],
                "description": "Flag indicating whether the incident involved a mental health component"
            },
            "priority_level": {
                "bsonType": "string",
                "enum": ["Priority 0", "Priority 1"],
                "description": "Urgency level — only Priority 0 and Priority 1 calls are included in this dataset"
            },
            "response_datetime": {
                "bsonType": "string",
                "description": "Date and time the unit was dispatched in YYYY-MM-DD HH:MM:SS format"
            },
            "response_hour": {
                "bsonType": "int",
                "minimum": 0,
                "maximum": 23,
                "description": "Hour of day the incident was reported, 0-23"
            },
            "initial_problem_category": {
                "bsonType": "string",
                "description": "Incident category as reported by the caller"
            },
            "final_problem_category": {
                "bsonType": "string",
                "description": "Incident category as ultimately classified by responding officers"
            },
            "number_of_units_arrived": {
                "bsonType": ["double", "null"],
                "description": "Total number of public safety units that responded to the incident"
            },
            "unit_time_on_scene": {
                "bsonType": ["double", "null"],
                "description": "Total time in seconds units spent on scene"
            },
            "response_time": {
                "bsonType": ["double", "null"],
                "description": "Time in seconds from call received to unit arrival on scene"
            },
            "officer_injured_killed_count": {
                "bsonType": "int",
                "description": "Number of officers injured or killed during the incident"
            },
            "subject_injured_killed_count": {
                "bsonType": "int",
                "description": "Number of subjects injured or killed during the incident"
            },
            "geo_id": {
                "bsonType": ["double", "null"],
                "description": "Full FIPS geographic identifier for the census block"
            },
            "census_block_group": {
                "bsonType": ["double", "null"],
                "description": "Census block group identifier for the incident location"
            },
            "date": {
                "bsonType": "string",
                "description": "Date of the incident in YYYY-MM-DD format"
            },
            "temperature_2m_max": {
                "bsonType": ["double", "null"],
                "description": "Maximum daily temperature in Fahrenheit at 2 meters above ground"
            },
            "temperature_2m_min": {
                "bsonType": ["double", "null"],
                "description": "Minimum daily temperature in Fahrenheit at 2 meters above ground"
            },
            "precipitation_sum": {
                "bsonType": ["double", "null"],
                "description": "Total daily precipitation in inches"
            },
            "windspeed_10m_max": {
                "bsonType": ["double", "null"],
                "description": "Maximum daily wind speed in mph at 10 meters above ground"
            },
            "weathercode": {
                "bsonType": ["int", "double", "null"],
                "description": "WMO weather interpretation code — 0 is clear sky, higher values indicate worse weather"
            },
            "day_of_week": {
                "bsonType": "int",
                "minimum": 0,
                "maximum": 6,
                "description": "Day of week as integer — 0 is Monday, 6 is Sunday"
            },
            "month": {
                "bsonType": "int",
                "minimum": 1,
                "maximum": 12,
                "description": "Month of the incident"
            },
            "day_of_year": {
                "bsonType": "int",
                "minimum": 1,
                "maximum": 366,
                "description": "Day of year — 1 to 366"
            },
            "is_weekend": {
                "bsonType": "int",
                "enum": [0, 1],
                "description": "1 if the incident occurred on a Saturday or Sunday, 0 otherwise"
            },
            "season": {
                "bsonType": "string",
                "enum": ["spring", "summer", "fall", "winter"],
                "description": "Meteorological season based on month"
            },
            "is_holiday": {
                "bsonType": "int",
                "enum": [0, 1],
                "description": "1 if the incident occurred on a US federal holiday in Texas, 0 otherwise"
            },
            "is_ut_game": {
                "bsonType": "int",
                "enum": [0, 1],
                "description": "1 if the incident occurred on a UT Longhorns home football game day, 0 otherwise"
            },
            "is_sxsw": {
                "bsonType": "int",
                "enum": [0, 1],
                "description": "1 if the incident occurred during the SXSW festival, 0 otherwise"
            },
            "is_acl": {
                "bsonType": "int",
                "enum": [0, 1],
                "description": "1 if the incident occurred during Austin City Limits festival, 0 otherwise"
            },
            "is_ut_movein": {
                "bsonType": "int",
                "enum": [0, 1],
                "description": "1 if the incident occurred during UT Austin move-in weekend, 0 otherwise"
            },
            "is_spring_break": {
                "bsonType": "int",
                "enum": [0, 1],
                "description": "1 if the incident occurred during UT Austin spring break, 0 otherwise"
            },
            "is_graduation": {
                "bsonType": "int",
                "enum": [0, 1],
                "description": "1 if the incident occurred during UT Austin graduation weekend, 0 otherwise"
            },
            "split": {
                "bsonType": "string",
                "enum": ["train", "test"],
                "description": "Model split designation — train is 2025 data, test is 2024 data"
            }
        }
    }
}
```

### Data Summary

## Data Summary

| Metric | Value |
|---|---|
| Total Documents | 17,399 |
| Date Range | 2024-01-01 to 2025-12-31 |
| Training Split (2025) | 8,453 calls |
| Test Split (2024) | 8,946 calls |
| Unique Incidents | 17,399 |
| Priority 0 Calls | 8,139 (46.8%) |
| Priority 1 Calls | 9,260 (53.2%) |
| Mental Health Incidents | 3,029 (17.4%) |
| Non-Mental Health Incidents | 14,370 (82.6%) |
| Avg Daily Call Count | ~23 calls/day |
| Min Daily Call Count | 10 |
| Max Daily Call Count | 40 |
| Avg Response Time | 691 seconds (~11.5 min) |
| Median Response Time | 536 seconds (~9 min) |
| Max Response Time | 25,873 seconds (~7.2 hrs) |
| Null Response Times | 102 |
| Avg Units Per Incident | 2.6 |
| Max Units Per Incident | 37 |
| Top Call Category | Welfare Check (3,831) |
| 2nd Top Call Category | Traffic Stop/Hazard (2,585) |
| 3rd Top Call Category | Disturbance (2,189) |
| Temperature Range | 31.0°F to 104.8°F |
| Max Single Day Precipitation | 5.01 inches |
| UT Home Game Days Covered | 14 (6 in 2025, 8 in 2024) |
| SXSW Days Covered | 18 (9 per year) |
| ACL Days Covered | 12 (6 per year) |
| Weather Source | Open-Meteo Historical API |
| Geographic Scope | Baker Sector, Austin TX |

### Data Dictionary

| Field | Type | Description | Values / Units |
|---|---|---|---|
| `_id` | int | Unique incident number, primary key | APD incident ID |
| `incident_type` | string | How the incident was initiated | "Dispatched Incident" |
| `mental_health_flag` | string | Whether incident involved mental health | "Mental Health Incident" / "Not Mental Health Incident" |
| `priority_level` | string | Urgency level assigned by dispatcher | "Priority 0", "Priority 1" |
| `response_datetime` | string | Date and time unit was dispatched | YYYY-MM-DD HH:MM:SS |
| `response_hour` | int | Hour of day incident was reported | 0–23 |
| `initial_problem_category` | string | Incident category as reported by caller | e.g. Welfare Check, Disturbance |
| `final_problem_category` | string | Incident category as classified by officers | e.g. Welfare Check, Disturbance |
| `number_of_units_arrived` | double | Number of units that responded | Count, occasional nulls |
| `unit_time_on_scene` | double | Time units spent on scene | Seconds |
| `response_time` | double | Time from call to unit arrival | Seconds, 102 nulls |
| `officer_injured_killed_count` | int | Officers injured or killed | Count, mostly 0 |
| `subject_injured_killed_count` | int | Subjects injured or killed | Count, mostly 0 |
| `geo_id` | double | Full FIPS geographic identifier | Census FIPS code |
| `census_block_group` | double | Census block group identifier | Census code |
| `date` | string | Date of incident | YYYY-MM-DD |
| `temperature_2m_max` | double | Max daily temperature | Fahrenheit |
| `temperature_2m_min` | double | Min daily temperature | Fahrenheit |
| `precipitation_sum` | double | Total daily precipitation | Inches |
| `windspeed_10m_max` | double | Max daily wind speed | MPH |
| `weathercode` | int | WMO weather code | 0 = clear, higher = worse weather |
| `day_of_week` | int | Day of week | 0 = Monday, 6 = Sunday |
| `month` | int | Month of incident | 1–12 |
| `day_of_year` | int | Day of year | 1–366 |
| `is_weekend` | int | Whether incident occurred on weekend | 0 or 1 |
| `season` | string | Meteorological season | spring, summer, fall, winter |
| `is_holiday` | int | Whether incident occurred on TX federal holiday | 0 or 1 |
| `is_ut_game` | int | Whether incident occurred on UT home football game day | 0 or 1 |
| `is_sxsw` | int | Whether incident occurred during SXSW | 0 or 1 |
| `is_acl` | int | Whether incident occurred during ACL festival | 0 or 1 |
| `is_ut_movein` | int | Whether incident occurred during UT move-in weekend | 0 or 1 |
| `is_spring_break` | int | Whether incident occurred during UT spring break | 0 or 1 |
| `is_graduation` | int | Whether incident occurred during UT graduation | 0 or 1 |
| `split` | string | Model split designation | "train" (2025), "test" (2024) |

### Numerical Feature Uncertainty

| Feature | Count | Mean | Std Dev | Min | 25% | Median | 75% | Max | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `response_time` | 17,297 | 691s | 744s | 11s | 352s | 536s | 806s | 25,873s | 102 nulls, right-skewed |
| `number_of_units_arrived` | 17,399 | 2.64 | 1.65 | 1 | 2 | 2 | 3 | 37 | Occasional large incidents |
| `unit_time_on_scene` | 17,399 | 9,305s | 15,993s | 2s | 2,027s | 5,109s | 10,898s | 696,475s | Highly right-skewed, outliers present |
| `officer_injured_killed_count` | 17,399 | 0.00 | 0.01 | 0 | 0 | 0 | 0 | 1 | Rare event, near-zero variance |
| `subject_injured_killed_count` | 17,399 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0 | No occurrences in dataset |
| `response_hour` | 17,399 | 13.09 | 6.84 | 0 | 8 | 14 | 19 | 23 | Peaks in afternoon/evening |
| `day_of_week` | 17,399 | 3.05 | 1.99 | 0 | 1 | 3 | 5 | 6 | Roughly uniform across week |
| `month` | 17,399 | 6.52 | 3.40 | 1 | 4 | 7 | 9 | 12 | Full year coverage both years |
| `day_of_year` | 17,399 | 183.28 | 103.85 | 1 | 95 | 183 | 273 | 366 | Full year coverage both years |
| `temperature_2m_max` | 17,399 | 81.6°F | 13.6°F | 31.0°F | 73.7°F | 84.6°F | 91.9°F | 104.8°F | Austin summers dominate distribution |
| `temperature_2m_min` | 17,399 | 62.5°F | 13.5°F | 17.4°F | 53.6°F | 66.2°F | 73.7°F | 82.4°F | Correlated with max temp |
| `precipitation_sum` | 17,399 | 0.12 in | 0.34 in | 0.00 in | 0.00 in | 0.00 in | 0.06 in | 5.01 in | Majority of days have no rain |
| `windspeed_10m_max` | 17,399 | 11.4 mph | 3.7 mph | 3.9 mph | 8.8 mph | 11.0 mph | 13.5 mph | 27.0 mph | Relatively low variance |
| `weathercode` | 17,399 | 29.6 | 27.4 | 0 | 3 | 51 | 53 | 73 | WMO code, bimodal distribution |


