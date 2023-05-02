# import data manipulation libraries
import numpy as np
import pandas as pd
# import visualization tools
import matplotlib.pyplot as plt
import seaborn as sns
# import statistical tools
from scipy import stats
# import data wrangle functions
import wrangle_austin_shelter as w

#----------------------------------------------------------------

def check_chi2_hypothesis(data, x, y, α=0.05):
    '''
    This function will check the provided x and y variables from the 
    provided dataset (data) for statistical relevence according 
    to a chi-square test (this is changable by entering the desired test as a kwarg)
    '''
    # run the requested statistical test on variables x and y from data
    observed = pd.crosstab(data[x], data[y])
    chi2, p, _, hypothetical = stats.chi2_contingency(observed)
    # if the resulting p-value is less than alpha, then reject the null hypothesis
    if p < α:
        # print results rejecting null hypothesis
        print(f"Since the p-value is less than {α}, \n\
we can reject the null hypothesis and conclude that {x} and {y} are not independent.")
        print(f"The chi-squared coefficient between \
{x} and {y} is {chi2:.2f} with a p-value of {p:.4f}")
        print('_______________________________________________________')
    # if p-value >= alpha, then we fail to reject the null hypothesis
    else:
        # print the results failing to reject the null hypothesis
        print(f"Since the p-value is greater than or equal to {α}, \n\
we fail to reject the null hypothesis and conclude \n\
that {x} and {y} are independent.")
        print('_______________________________________________________')

#----------------------------------------------------------------

def get_plot_outcomes(animals):
    '''
    display a plot showing the total amount of outcomes by type
    '''
    # create the bar plot
    animals.outcome.value_counts().plot.bar()
    # add a title
    plt.title('Adoption Accounts for 69% of Outcomes', size=20)
    # add x and y axis labels
    plt.xlabel('Outcome of Trip to Animal Shelter', size=18)
    plt.ylabel('Number of Animals with Outcome', size = 18)
    # increase the size of the xticks
    plt.xticks(size=16)
    # display the plot
    plt.show()

#----------------------------------------------------------------

def get_plot_deaths_by_year(train):
    '''
    display a plot of total euthenization, deaths and disposal outcomes over time
    '''
    # create a subset of the data containing death outcomes
    deaths = train[train.outcome == 'death']
    # set the index to the date of the outcome
    deaths = deaths.set_index('datetime_out')
    # resample the data by year, and create the plot
    deaths.resample('Y').animal_id.count().plot()
    # add a title
    plt.title('Deaths by Year', size=16)
    # add axis labels
    plt.xlabel('Year', size=15)
    plt.ylabel('Number of Animal Deaths in Shelter', size=15)
    # display the plot
    plt.show()

#----------------------------------------------------------------

def get_plot_wildlife_deaths(animals):
    '''
    display a plot showing the outcomes for animals with intake_type of 'Wildlife'
    '''
    # create a data subset of only animals with intake type of 'wildlife'
    wildlife = animals[animals.intake_type == 'Wildlife']
    # create a histogram of the data
    sns.histplot(data = wildlife, x='outcome')
    # add a title
    plt.title('Outcomes for Wildlife', size =18)
    # add axis labels
    plt.xlabel('Outcome of Trip to Animal Shelter', size= 16)
    plt.ylabel('Number of Animals', size=16)
    # change the tick size
    plt.xticks(size=15)
    plt.yticks(size=15)
    # display the plot
    plt.show()

#----------------------------------------------------------------

def get_plot_named_vs_unnamed(train):
    '''
    display 2 plots showing the difference in outcomes for animals with a name
    and animals without a name
    '''
    # create data subsets of animals with names and animals without names
    named = train[train.has_name == True].sort_values('outcome')
    unnamed = train[train.has_name == False].sort_values('outcome')
    # create a figure
    plt.figure(figsize=(12,8))
    # create a subplot
    plt.subplot(1,2,1)
    # create the first histogram
    sns.histplot(data = named, x='outcome')
    # add a title for the named plot
    plt.title('Named Animal Outcomes', size =18)
    # add axis labels
    plt.xlabel('Outcome of Trip to Animal Shelter', size= 16, loc='right')
    plt.ylabel('Number of Animals', size=18)
    # change the tick sizes
    plt.xticks(size=15)
    plt.yticks(size=15)
    # create the second subplot
    plt.subplot(1,2,2)
    # create the histogram for unnamed animal outcomes
    sns.histplot(data = unnamed, x='outcome')
    # add a title
    plt.title('Unnamed Animal Outcomes', size =18)
    # remove the axis labels for the second plot
    plt.xlabel('')
    plt.ylabel('')
    # change tick size
    plt.xticks(size=15)
    plt.yticks(size=15)
    # display the plots
    plt.show()

#----------------------------------------------------------------

def get_plot_outcomes_by_day(train):
    '''
    display a plot showing the total number of outcomes per day of the week
    '''
    # create variables storing day of week and day name
    train['outcome_day'] = train.datetime_out.dt.day_name()
    train['weekday_num'] = train.datetime_out.dt.day_of_week
    # create the area plot
    train.groupby('outcome_day').outcome.value_counts().unstack().plot.area()
    # add a title
    plt.title('Outcome Types by Day of Week', size=20)
    # add axis labels
    plt.xlabel('Day of Week', size=16)
    plt.ylabel('Number of Animal Outcomes', size=16)
    # display the plot
    plt.show()

#----------------------------------------------------------------

def get_plot_outcomes_by_sex(train):
    '''
    display a plot showing the difference in outcomes based on sex_upon_outcome
    '''
    # create the plot
    sns.histplot(data=train, x='outcome', hue='sex_upon_outcome', stat='count')
    # add a title
    plt.title('Outcome Types by Sex Upon Outcome', size=20)
    # add axis labels
    plt.xlabel('Outcome', size=16)
    plt.ylabel('Number of Animal Outcomes', size=16)
    # display the plot
    plt.show()