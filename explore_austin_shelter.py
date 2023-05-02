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
    animals.outcome.value_counts().plot.bar()
    plt.title('Adoption Accounts for 69% of Outcomes', size=20)
    plt.xlabel('Outcome of Trip to Animal Shelter', size=18)
    plt.ylabel('Number of Animals with Outcome', size = 18)
    plt.xticks(size=16)
    plt.show()

#----------------------------------------------------------------

def get_plot_deaths_by_year(train):
    deaths = train[train.outcome == 'death']
    deaths = deaths.set_index('datetime_out')
    deaths.resample('Y').animal_id.count().plot()
    plt.title('Deaths by Year', size=16)
    plt.xlabel('Year', size=15)
    plt.ylabel('Number of Animal Deaths in Shelter', size=15)
    plt.show()

#----------------------------------------------------------------

def get_plot_wildlife_deaths(animals):
    wildlife = animals[animals.intake_type == 'Wildlife']
    sns.histplot(data = wildlife, x='outcome')
    plt.title('Outcomes for Wildlife', size =18)
    plt.xlabel('Outcome of Trip to Animal Shelter', size= 16)
    plt.ylabel('Number of Animals', size=16)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.show()

#----------------------------------------------------------------

def get_plot_named_vs_unnamed(train):
    named = train[train.has_name == True].sort_values('outcome')
    unnamed = train[train.has_name == False].sort_values('outcome')
    plt.figure(figsize=(12,8))
    plt.subplot(1,2,1)
    sns.histplot(data = named, x='outcome')
    plt.title('Named Animal Outcomes', size =18)
    plt.xlabel('Outcome of Trip to Animal Shelter', size= 16, loc='right')
    plt.ylabel('Number of Animals', size=18)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.subplot(1,2,2)
    sns.histplot(data = unnamed, x='outcome')
    plt.title('Unnamed Animal Outcomes', size =18)
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.show()

#----------------------------------------------------------------

def get_plot_outcomes_by_day(train):
    train['outcome_day'] = train.datetime_out.dt.day_name()
    train['weekday_num'] = train.datetime_out.dt.day_of_week
    train.groupby('outcome_day').outcome.value_counts().unstack().plot.area()
    plt.title('Outcome Types by Day of Week', size=20)
    plt.xlabel('Day of Week', size=16)
    plt.ylabel('Number of Animal Outcomes', size=16)
    plt.show()

#----------------------------------------------------------------

def get_plot_outcomes_by_sex(train):
    sns.histplot(data=train, x='outcome', hue='sex_upon_outcome', stat='count')
    plt.title('Outcome Types by Sex Upon Outcome', size=20)
    plt.xlabel('Outcome', size=16)
    plt.ylabel('Number of Animal Outcomes', size=16)
    plt.show()