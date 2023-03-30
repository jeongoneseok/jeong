#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load pyspark, pandas
from pyspark import SparkConf, SparkContext
import pandas as pd


# In[2]:


# Configure Spark
conf = SparkConf().setMaster("local").setAppName("uber-date-trips")
sc = SparkContext.getOrCreate();


# In[3]:


# Dataset sample
filename = "fhvhv_tripdata_2020-03_short.csv"


# In[4]:


# Data parsing
lines = sc.textFile("./data/" + filename)   # .csv -> RDD object
header = lines.first()
filtered_lines = lines.filter(lambda row:row != header) # all lines excepting the header


# In[5]:


"""
    Run the Spark job
    - map(): apply the transformation on every element of RDD -> new RDD
    - countByValue(): action that returns the count of each unique value
    - x.split(", ")[2]: extract the value of pickup_datetime from a row
    e.g., 2020-03-01 00:03:40
    - .split(" ")[0]: extract the date from the pickup_datetime
    e.g., 2020-03-01
"""

dates = filtered_lines.map(lambda x: x.split(",")[0].split(" ")[0])
result = dates.countByValue()


# In[6]:


# Save results as a csv file
pd.Series(result, name="trips").to_csv("./data/trips_date.csv")


# In[7]:


# Visualize the results
import matplotlib.pyplot as plt

trips = pd.read_csv("./data/trips_date.csv")
trips.plot()
plt.show()


# In[ ]:





# In[ ]:




