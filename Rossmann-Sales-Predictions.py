
# coding: utf-8

# In[29]:

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt

#importing the training data
rossmann= pd.read_csv("train.csv", dtype = "unicode")

#printing descriptive statistics of different variables
rossmann.describe(percentile_width=None, percentiles=None, include=None, exclude=None)

rossmann = rossmann.loc[rossmann['Sales'] > 0]

rossmann["Sales"]=rossmann["Sales"].astype(float)
rossmann["Customers"]=rossmann["Customers"].astype(float)

m, b = np.polyfit(rossmann["Customers"], rossmann["Sales"], 1)

predictions=m*rossmann["Customers"]+b
sales_mean=np.mean(rossmann["Sales"])
sumerror=np.mean((sales_mean-predictions)**2)
stddev=np.sqrt(sumerror)
print ("Sales = %s * Customers + %s" % (m,b))
print (predictions.head(5))
print (stddev)

plt.plot(rossmann["Customers"].head(150), rossmann['Sales'].head(150), 'bs')
plt.plot(rossmann["Customers"].head(150), m*rossmann["Customers"].head(150)+b, 'r--')
plt.title("Rossmann Sales vs Customers")
plt.xlabel("Customers")
plt.ylabel("Sales")
plt.show()

sales=rossmann["Sales"].head(150)
customers=rossmann["Customers"].head(150)

data_to_plot=[sales, customers]
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.set_title('Rossmann Sales and Customers')
ax.set_ylabel('Value')
bp = ax.boxplot(data_to_plot)
plt.show()

