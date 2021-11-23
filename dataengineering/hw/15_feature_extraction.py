import pandas as pd
import numpy as np
from pandas import datetime
from matplotlib import pyplot as plt

"""
Load AirQualityUCI Data
"""

def parser(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

input_file = './data/AirQualityUCI_refined.csv'

df = pd.read_csv(input_file,
                 index_col=[0],
                 parse_dates=[0],
                 date_parser=parser)

df.head()


# Visualization setup
%matplotlib
%config InlineBackend.figure_format = 'svg'

from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]
plt.ion() # enable the interactive mode

import seaborn as sns
sns.set()  # set plot styles


# Visualize the 'CO(GT)' column
df['CO(GT)'].interpolate(inplace=True)


"""
Binning
"""

max_val = df['CO(GT)'].max()
min_val = df['CO(GT)'].min()
print(max_val, min_val)

# Make interval values
bins = np.linspace(min_val, max_val, 6)
bins

# Labels for each bin
labels=['0 <=x<2.38', '2.38<=x<4.76', '4.76<=x<7.14',
       '7.14<=x<9.52', '9.52<=x<11.9']


# Convert the numerical values into the categorical values
df['bins'] = pd.cut(df['CO(GT)'], bins=bins,
                    labels=labels, include_lowest=True)
df.info()

# Print bins
df['bins'][:50]


# Visualize the histogram of bins
plt.hist(df['bins'], bins=5)
plt.show()


"""
Log Transform
"""

# Distribution of original data
sns.distplot(df['PT08.S3(NOx)'])


# Calculate natural logarithm on 'CO(GT)' column
df['log'] = np.log10(df['PT08.S3(NOx)'])


# Min values for each column
df.min()


# Distribution after log transform
sns.distplot(df['log'])
plt.xlabel('log(PT08.S3(NoX))')
plt.show()


"""
One-hot Encoding
"""

# Make a dataset

emp_id = pd.Series([1, 2, 3, 4, 5])
gender = pd.Series(['Male', 'Female', 'Female', 'Male', 'Female'])
remarks = pd.Series(['Nice', 'Good', 'Great', 'Great', 'Nice'])

df_emp = pd.DataFrame()
df_emp['emp_id'] = emp_id
df_emp['gender'] = gender
df_emp['remarks'] = remarks

# Print DataFrame
df_emp

# Print unique values for each column
print(df_emp['emp_id'].unique())
print(df_emp['gender'].unique())
print(df_emp['remarks'].unique())

# One-hot encoding the categorial values
df_emp_encoded = pd.get_dummies(
    df_emp, columns = ['gender', 'remarks'])

df_emp_encoded


"""
Normalization
"""

# Visualize two columns of different scales
plt.plot(df['CO(GT)'], label='CO')
plt.plot(df['PT08.S2(NMHC)'], label='NMHC')
plt.legend(loc='best')


# Normalize the 'CO(GT)' column
co = df['CO(GT)'].copy()
co_max = co.max()
co_min = co.min()

df['CO_Norm'] = (co - co_min) / (co_max - co_min)
df['CO_Norm']


# Normalize the 'PT08.S2(NMHC)' column
nmhc = df['PT08.S2(NMHC)'].copy()
nmhc_max = nmhc.max()
nmhc_min = nmhc.min()

df['NMHC_Norm'] = (nmhc - nmhc_min) / (nmhc_max - nmhc_min)
df['NMHC_Norm']


# Visualized normalized columns
plt.plot(df['CO_Norm'], label='CO (normalized)')
plt.plot(df['NMHC_Norm'], label='NMHC (normalized)')
plt.legend(loc='best')


"""
Feature Split
"""

# Make untidy movie data
movies = pd.Series(["The Godfather, 1972, Francis Ford Coppola",
                    "Contact, 1997, Robert Zemeckis",
                   "Parasite, 2019, Joon-ho Bong"])

movies


# Divide movie data into title, year, director columns

lst_title = []
lst_year = []
lst_director = []

for val in movies:
    title, year, director = val.split(',')  # data split
    lst_title.append(title)
    lst_year.append(year)
    lst_director.append(director)

print(lst_title)
print(lst_year)
print(lst_director)


# Make a DataFrame object
df_movie = pd.DataFrame()
df_movie['title'] = lst_title
df_movie['year'] = lst_year
df_movie['director'] = lst_director

df_movie
