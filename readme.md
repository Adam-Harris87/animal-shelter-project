# Austin Animal Shelter Analysis


# Project Description

There are tens of thousands of animals in animal shelters around the country. In this project, we will be attempting to identify some of the factors that can predict if an animal will be adopted. Our hope is that by using this data, animal shelters will be able to better identify which animals to arrange for transfer, or focus adoption efforts on so that less animals might die in the shelter.


# Project Goal

With this project we will attempt to identify some of the factors that help lead to an animal being adopted from an animal shelter. We will also build a machine learning model with the intent of identifying animals with a lower chance to being adopted in order to get them a little extra help in getting adopted.


# Initial Hypotheses

- What is the total percentage of animals that are adopted?
- Did the amount of animals euthenized at the shelter increase durring covid-19 years?
- What is the likelyhood that wildlife will leave the shelter?
- Are certain days busier than others for the animal shelter?
- Does an animal having a name affect the chances of them being adopted?
- Does an animal's sex have an affect on the adoption chance?


# Project Plan

- Planning - The steps required to be taken during this project will be laid out in this readme file.

- Acquisition - The data was acquired via api from two source datasets containing intake and outcomes for animals at the Austin, TX animal shelter. The data is located at: https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Intakes/wter-evkm (intake data) and https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Outcomes/9t4d-g238 (outcome data). Once the data was acquired, it was combined into one dataset using a merge function.

- Preparation - Redundant columns resulting from merging the datasets were removed. Column names were changed to be easier to work with. The data type for date columns was changed to datetime. A new feature was created with a true/false value to identify if the animal has a name. Similar outcome types were grouped together to be easier to work. Categorical columns were encoded into integers/dummy variables. Null values in the outcome_subtype column were filled with 'None' since we were only conserned with the main outcome type at this time. Rows with null values in the outcome_type column were removed, since these may be animals that are still at the shelter and not useful for modeling purposes. Data was split into train, validate and test (56%, 24%, 20%) groups to prevent overfiting.

- Exploration - We created multiple visulizations to identify patterns in the data. We also performed some statistical tests to confirm if suspected patterns were statistically relevent. We also looked for variables that could be better understood by converting the variable into bins.

- Modeling - In this project we attempted to classify if an animal will be adopted, therefore we used classification models for this problem. For our assessment metrics we were looking to be overall correct in our predictions as much as possible, along with missing as few animals that have a low chance of being adopted. We will be looking to maximize the accuracy and recall metrics. Our baseline model (saying that an animal will be adopted) is accurate 69% of the time. Our models used the features: has_name, sex_upon_outcome, intake_type, and animal_type

- Delivery - We will be packaging our findings in a final report python notebook. Results will be posted on GitHub.


# Data Dictionary

| Field 		   |        Data type 		|				Description				       |
|------------------|------------------------|----------------------------------------------|
| animal_id        |                  object| unique id for each animal 				   |
| datetime_in      |          datetime64[ns]| day and time that the animal arrived	       |
| found_location   |                  object| address that stray was found				   |
| intake_type      |                  object| reason why the animal arrived at the shelter |
| intake_condition |                  object| medical condition how the animal arrived	   |
| animal_type      |                  object| if the animal is a dog, cat, bird, etc 	   |
| sex_upon_intake  |                  object| sex of the animal when arriving			   |
| age_upon_intake  |                  object| age when the animal arrived at the shelter   |
| breed            |                  object| type of dog, cat, etc of the animal 		   |
| color            |                  object| color of the animal's fur 				   |
| name             |                  object| the animal's name, if it has one 			   |
| datetime_out     |          datetime64[ns]| day and time the animal left the shelter 	   |
| date_of_birth    |          datetime64[ns]| day the animal was born					   |
| outcome_type     |                  object| what way the animal left the shelter 		   |
| sex_upon_outcome |                  object| animal's sex when leaving the shelter 	   |
| age_upon_outcome |                  object| age of the animal when it left the shelter   |
| outcome_subtype  |                  object| more specific outcome information 		   |

# Steps to Reproduce

To recreate our findings, you will need the final_report.ipynb file along with all .py files from the GitHub repository stored in the same directory on your device. The dataset will be downloaded via api when running the final_report file. Your result numbers may differ slightly from our numbers since your downloaded data will be more current than our data which was downloaded on 4-27-2023.