# import data manipulation libraries
import numpy as np
import pandas as pd
# import file acquisition tools
import os
import requests
# import splitting function
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#-----------------------------------------------

def get_data(url, offset=1000):
    '''
    this function will return a dataFrame with all of the results from the passed url
    until the last page of results
    '''
    print('downloading data via api')
    # create request.get for passed url
    page = requests.get(url)
    # create a dataframe withe the first page results
    df = pd.DataFrame(page.json())
    # set first offset amount
    x=offset
    # loop through all pages of results for passed url
    while page.json():
        # set the page url to the next page url
        next_url = url + f'&$offset={x}'
        page = requests.get(next_url)
        df = pd.concat([df, pd.DataFrame(page.json())], axis=0)
        x += offset
        print(f'{x}')
        if x >= 170000:
            break
    # display how many rows were downloaded
    print(f'{df.shape[0]} records were downloaded via api')
    # return the datframe
    return df

#-----------------------------------------------

def acquire_austin_animal_shelter_data():
    '''
    This function will read in austin animal shelter intake and outcome data via csv,
    it will then rename to the column names to lowercase and remove spaces,
    it will then combine the csv files into one DataFrame and return the combined df.
    '''
    # set the filenames we are looking for
    intake_filename = 'Austin_Animal_Center_Intakes.csv'
    outcome_filename = 'Austin_Animal_Center_Outcomes.csv'
    # set the api paths
    token = 'Fq34fTdVZDfsd8JcA1Vq4Qf1s'
    intake_api = f'https://data.austintexas.gov/resource/wter-evkm.json?$$app_token={token}'
    outcome_api = f'https://data.austintexas.gov/resource/9t4d-g238.json?$$app_token={token}'
    
    # check if intake and outcome files exist in the local directory
    if os.path.exists(intake_filename):
        # read in csv files for intakes for animals
        print('reading intake data from local file')
        intakes = pd.read_csv(intake_filename)
        # if the file was downloaded from webpage it will not have 'unnamed:_0' column
        # but if the file was created by function it will have 'unnamed:_0' column
        if 'Unnamed:_0' in intakes.columns.to_list():
            intakes.drop(columns='Unnamed:_0', inplace=True, index_col=0)
    else:
        # if the intake file does not exist locally, then download the data via api
        intakes = get_data(intake_api)
        intakes.datetime = pd.to_datetime(intakes.datetime)
        intakes.to_csv(intake_filename)
        
    # check if outcome file exist in the local directory
    if os.path.exists(outcome_filename):
        # read in csv files for outcomes for animals
        print('reading outcome data from local file')
        outcomes = pd.read_csv(outcome_filename, index_col=0)
        # if the file was downloaded from webpage it will not have 'unnamed:_0' column
        # but if the file was created by function it will have 'unnamed:_0' column
        if 'Unnamed:_0' in outcomes.columns.to_list():
            outcomes.drop(columns='Unnamed:_0', inplace=True)
    else:
        # if the intake file does not exist locally, then download the data via api
        outcomes = get_data(outcome_api)
        outcomes.datetime = pd.to_datetime(outcomes.datetime)
        outcomes.drop(columns='monthyear', inplace=True)
        outcomes.to_csv(outcome_filename)
        
    # rename column names to lowercase and remove spaces
    intakes.columns = intakes.columns.str.lower().str.replace(' ','_').to_list()
    outcomes.columns = outcomes.columns.str.lower().str.replace(' ','_').to_list()
    # merge dataframes into one
    animals = pd.merge(left=intakes, right=outcomes, how='inner', 
                       on='animal_id', suffixes=('_in','_out'))
    
    # return the combined dataframe
    return animals

#-----------------------------------------------

def prepare_austin_animal_shelter(animals):
    '''
    this function will prepare the animal shelter data by changing the datetime_in and out
    into datatime dtype, it will then drop the redundant color_out, breed_out, name_out
    and animal_type_out columns and rename the remaining columns, it will then
    return the modified dataframe.
    '''
    # change dtypes to datetime
    animals['datetime_in'] = pd.to_datetime(animals.datetime_in)
    animals['datetime_out'] = pd.to_datetime(animals.datetime_out)
    animals['date_of_birth'] = pd.to_datetime(animals.date_of_birth)
    
    # remove redundant columns
    animals = animals.drop(columns=['color_out', 'breed_out', 
                                    'name_out', 'animal_type_out'])
    # drop redundaant columns for month_year
    if 'monthyear_in' in animals.columns.to_list():
        animals = animals.drop(columns='monthyear_in')
    elif 'monthyear_out' in animals.columns.to_list():
        animals = animals.drop(columns='monthyear_out')
    # rename remaining columns
    animals = animals.rename(columns={'color_in':'color', 
                                      'breed_in':'breed', 'name_in':'name',
                                      'animal_type_in':'animal_type'})
    
    # remove the 2 rows with nullsin the sex_upon_intake column, 
    # 1 of which is a test row
    animals = animals[animals.sex_upon_intake.isna() == False]
    
    # create a new column with binned outcome_type
    animals['outcome'] = np.where(animals.outcome_type == 'Adoption', 'adopted',
                          np.where(animals.outcome_type == 'Return to Owner', 'adopted',
                          np.where(animals.outcome_type == 'Transfer', 'transfered',
                          np.where(animals.outcome_type == 'Euthanasia', 'death',
                          np.where(animals.outcome_type == 'Rto-Adopt', 'adopted',
                          np.where(animals.outcome_type == 'Died', 'death',
                          np.where(animals.outcome_type == 'Disposal', 'death',
                          np.where(animals.outcome_type == 'Missing', 'transfered',
                          np.where(animals.outcome_type == 'Stolen', 'adopted',
                          np.where(animals.outcome_type == 'Relocate', 'transfered',
                          np.where(animals.outcome_type.isnull() == True, 'transfered', ''
                                  )))))))))))
    
    # create a column of bool values for if the animal has a name
    animals['has_name'] = np.where(animals.name.isna(), False, True)
    animals.name = animals.name.fillna('None')
    
    # fill null values in the outcome_subtype with 'None' since we alread have a 
    # main outcome type for these rows
    animals.outcome_subtype = animals.outcome_subtype.fillna('None')
    
    # drop the remaining null values (less than 100) for now, these are animals without
    # outcomes, which may mean they are still at the shelter
    animals = animals.dropna()


    # encode categorical columns so we can use them during explore and modeling
    # make a list of columns to encode
    encode_cols = ['intake_type', 'intake_condition', 'animal_type', 'sex_upon_intake',
               'breed', 'color', 'sex_upon_outcome', 'outcome_subtype']
    # create encoder object
    le = LabelEncoder()
    for col in encode_cols:
        le.fit(animals[col])
        # create a new column with the encoded values
        animals[f'{col}_encoded'] = le.transform(animals[col])
    # we want to one-hot encode the outcome variable since that is our target
    animals = pd.concat([animals, pd.get_dummies(animals.outcome)], axis=1)
    
    # return the modified dataframe
    return animals

#-----------------------------------------------

def split_austin_animal_shelter(df):
    '''
    This function splits a dataframe into 
    train, validate, and test in order to explore the data and to create and validate models. 
    It takes in a dataframe and contains an integer for setting a seed for replication. 
    Test is 20% of the original dataset. The remaining 80% of the dataset is 
    divided between valiidate and train, with validate being .30*.80= 24% of 
    the original dataset, and train being .70*.80= 56% of the original dataset. 
    The function returns, train, validate and test dataframes. 
    '''
    # Here we are spliting train into .8 of the original dataset. 
    # and test into 20% of the original dataset.
    train, test = train_test_split(df, test_size = .2, random_state=123)
    # here we assign validate to be .3 of the train dataset 
    train, validate = train_test_split(train, test_size=.3, random_state=123)
    # returns train validate and test dataframes
    return train, validate, test

#-----------------------------------------------

def wrangle_austin_animal_shelter():
    '''
    This function will perform acquisition, preparation and train, validate, test
    split into one function call and will return the cleaned dataframe along with 
    the train, validate and test dataframes.
    '''
    animals_filename = 'animal_shelter.csv'
    # check if a cached file of the dataset exists in the local directory
    # if os.path.exists(animals_filename):
    #     print('getting animal shelter data from local file')
    #     animals = pd.read_csv(animals_filename, index_col=0)
    # else:
        # acquire data from csv files
    animals = acquire_austin_animal_shelter_data()
    # prepare the data
    animals = prepare_austin_animal_shelter(animals)
    animals.to_csv(animals_filename)
    
    # remove 'unnamed:_0' column if it exists
    if 'unnamed:_0' in animals.columns.to_list():
        animals.drop(columns='unnamed:_0', inplace=True)
    # split data into train, validate and test groups
    train, validate, test = split_austin_animal_shelter(animals)
    # return all the data
    return animals, train, validate, test