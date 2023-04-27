import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#-----------------------------------------------

def acquire_austin_animal_shelter_data():
    '''
    This function will read in austin animal shelter intake and outcome data via csv,
    it will then rename to the column names to lowercase and remove spaces,
    it will then combine the csv files into one DataFrame and return the combined df.
    '''
    # read in csv files for intake and outcomes for animals
    intakes = pd.read_csv('Austin_Animal_Center_Intakes.csv')
    outcomes = pd.read_csv('Austin_Animal_Center_Outcomes.csv')
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
    # remove redundant columns
    animals = animals.drop(columns=['color_out', 'breed_out', 
                                    'name_out', 'animal_type_out'])
    # rename remaining columns
    animals = animals.rename(columns={'color_in':'color', 
                                      'breed_in':'breed', 'name_in':'name',
                                      'animal_type_in':'animal_type'})
    # drop redundaant columns for month_year
    animals = animals.drop(columns=['monthyear_in', 'monthyear_out'])
    # remove the 2 rows with nullsin the sex_upon_intake column, 
    # 1 of which is a test row
    animals = animals[animals.sex_upon_intake.isna() == False]
    # create a new column with binned outcome_type
    animals['outcome'] = np.where(animals.outcome_type == 'Adoption', 'adopted',
                          np.where(animals.outcome_type == 'Return to Owner', 'returned',
                          np.where(animals.outcome_type == 'Transfer', 'transfered',
                          np.where(animals.outcome_type == 'Euthanasia', 'death',
                          np.where(animals.outcome_type == 'Rto-Adopt', 'returned',
                          np.where(animals.outcome_type == 'Died', 'death',
                          np.where(animals.outcome_type == 'Disposal', 'death',
                          np.where(animals.outcome_type == 'Missing', 'unknown',
                          np.where(animals.outcome_type == 'Stolen', 'unknown',
                          np.where(animals.outcome_type == 'Relocate', 'transfered',
                          np.where(animals.outcome_type.isnull() == True, 'unknown', ''
                                  )))))))))))
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
    # acquire data from csv files
    animals = acquire_austin_animal_shelter_data()
    # prepare the data
    animals = prepare_austin_animal_shelter(animals)
    # split data into train, validate and test groups
    train, validate, test = split_austin_animal_shelter(animals)
    # return all the data
    return animals, train, validate, test