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



# Visualization setup
%matplotlib
from matplotlib import pyplot as plt
import seaborn; seaborn.set()  # set plot styles
%config InlineBackend.figure_format = 'svg'
plt.rcParams['figure.figsize'] = [10, 5]
plt.ion() # enable the interactive mode

# Visualize the 'CO(GT)' variable
df['CO(GT)'].plot()

# Linear interpolation
co = df['CO(GT)'].copy()
co.interpolate(inplace=True)

# Visualize original and imputed data
plt.plot(df['CO(GT)'], label='original', zorder=2)
plt.plot(co, label='linear interpolation', zorder=1)
plt.legend(loc='best')

# Detecting outliers using Boxplot
plt.boxplot(co)
plt.title("Detecting outliers using Boxplot")
plt.xlabel('CO(GT)')

# Calculate correlations between variables
corr_matrix = df.corr()
print(corr_matrix)

# Choose the least correlated variable
rh = df['RH'].copy().interpolate() # RH(Relative Humidity)

# Visualize a scatter plot(CO, RH)
plt.scatter(co, rh, s=12, c='black')
plt.xlabel('CO(GT)')
plt.ylabel('RH')

# Choose the most correlated variable
nmhc = df['PT08.S2(NMHC)'].copy().interpolate() # NMHC(non-methane hydrocarbons)

# Visualize a scatter plot(CO, NMHC)
plt.scatter(co, nmhc, s=12, c='black')
plt.xlabel('CO(GT)')
plt.ylabel('PT08.S2(NMHC)')


"""
IQR-based Outlier Detection
"""

# Q1, Q2(median), Q3
q1 = co.quantile(0.25)
median = co.quantile(0.5)
q3 = co.quantile(0.75)
print(q1, median, q3)

# IQR, upper_fence, lower_fence
iqr = q3-q1
upper_fence = q3 + 1.5*iqr
lower_fence = q1 - 1.5*iqr
print(upper_fence, lower_fence)

# Filtering the outliers
outliers = co.loc[(co > upper_fence) | (co < 0)]
print(outliers)

# Mask for outliers
mask = co.index.isin(outliers.index)
mask

# Visualize the normal data and outliers
plt.plot(co[~mask], label='normal', color='blue',
    marker='o', markersize=3, linestyle='None')
plt.plot(outliers, label='outliers', color='red',
    marker='x', markersize=3, linestyle='None')
plt.legend(loc='best')

# Removing the outliers
co_refined = co.copy()
co_refined[mask] = np.nan
print(co_refined[mask])
co_refined.plot()

# Linear interpolation for reconstructing outliers removed.
co_refined.interpolate(inplace=True)
co_refined.plot()


"""
Detecting Outliers with Z-Scores
"""

# Visualize the distribution of the 'CO(GT)' variable
import seaborn as sns
sns.distplot(co)

# Mean, Standard deviation
mean = np.mean(co)
std = np.std(co)
print(mean, std)

# Calculate Z-scores for each data points

outliers = []
thres = 3   # Z-score threshold

for i in co:
    z_score = (i-mean) / std
    if (np.abs(z_score) > thres):
        print(z_score)
        outliers.append(i)


# Simplified version of filtering outliers
outliers = co.loc[np.abs((co-mean)/std) > 3].copy()
outliers

# Mask for outliers
mask = co.index.isin(outliers.index)
mask

# Comparison of distributions before/after outlier removal
sns.distplot(co, axlabel='CO(GT)', label='original')
sns.distplot(co[~mask], label='outliers removed')
plt.legend(loc='best')

# [exer] Adjust thres

# Flooring and Capping
floor = co.quantile(0.1)
cap = co.quantile(0.9)

co.loc[co < floor] = floor
co.loc[co > cap] = cap

# Visualize the result
co.plot()
