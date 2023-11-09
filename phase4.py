# importing required libraries
import pandas as pd
import numpy as np
import cmap
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor



#loading the dataset
data = pd.read_csv("C:\\Users\\CT\\Desktop\\PoductDemand.csv")
print(data)
# data.isnull().sum()
# data = data.dropna()
# print(data)





#FEATURE ENGINEERING

# Data visualization
symbol = ['square']
plot = px.scatter(data, x="Units Sold", y="Total Price",
                 size='Units Sold',symbol_sequence = symbol)
plot.update_traces(marker=dict(color='purple'))
plot.show()


# Correlation betweeen the data
print(data.corr())


#Visualizing correlation
correlations = data.corr(method='pearson')
plt.figure(figsize=(20, 20))
cmap1=cmap.Colormap(['red','green','purple'])
sns.heatmap(correlations, annot=True,linecolor='black',linewidths=3)
plt.show()

#Algorithm for Model Building
x = data[["Total Price", "Base Price"]]
y = data["Units Sold"]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain.values, ytrain.values)

#PREDICTING THE DEMAND
features = np.array([[133.00, 140.00]])
prediction = model.predict(features)
print(prediction)