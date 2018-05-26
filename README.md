# K-Means Clustering of Major US Cities based on Climatic Conditions:Weather Data     

Authors:  **Sohit Reddy Kalluru** and **Akshay Sanjay Agrawal**

---



---

## Introduction
* The code is written for capturing live weather data for different cities across US.The attributes that are  considered are current temparature,maximum temparature,minimum temparature,humidity,wind speed,climatic conditions (Cloudy,Sunny,Partlycloudy,Rainy).

![Image of Plot](https://github.com/SohitKalluru/K-Means-Clustering-of-Major-US-Cities-based-on-Climatic-Conditions-Weather-Data-/blob/master/Images/final.PNG)

* The source of  weather data is from yahoo weather portal using a Yahoo-weather-API.The data can be imported in json and xml format.

* Source of data: https://developer.yahoo.com/weather/
* Data obtained is a live data and is updated on daily basis.

---

## Sources
  
- [Weather-api](https://github.com/AnthonyBloomer/weather-api) , the python wrapper for yahoo weather API ,is used in this code  for       accessing api link and retrieves data from json format.
- [Folium](http://folium.readthedocs.io/en/latest/quickstart.html), a powerful Python library that helps you create different types of Leaflet maps.
- WOEID(Where On Earth IDentifier),for US Cities is obtained from this [link](https://gist.github.com/lukemelia/353493).
- [Pricipal Component Analysis(PCA)]http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html ,a Linear       dimensionality reduction.
- [K-Means](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) for clustering.
- [DataFrame](https://pandas.pydata.org/pandas-docs/stable/10min.html) to convert api from json format to dataframe .
- [Outliers](http://www.whatissixsigma.net/box-plot-diagram-to-identify-outliers/) -  to standardize data obtained and to get more information about outliers.

**The code is executed in the Jupyter and spyder environment. If you're trying to execute using cmd, Windows Powershell or Apple command prompt the method may or may not vary.**

## Explanation of the Code

The code, `weather_data.py`, begins by importing necessary Python packages:

```
weather
Pandas
Folium
scikit
```
The  3 packages could be  installed as follows :
```
$ pip install folium
$ pip install weather-api
$ pip install pandas
$ pip install scikit-learn
```
- For more Information regarding the packages , refer from sources provided above.

Importing the packages after installation. 
```
import folium
from weather import Weather, Unit
import pandas as pd
import matplotlib.pyplot as plt
```


- Importing data from yahoo weather api using weather-api python wrapper. The Data is printed to verify:


```
weather = Weather(unit=Unit.CELSIUS)
datax=[]
for i in range(0,len(places)):
    location = weather.lookup(places[i])
    location.wind()['temp']=location.condition().temp()
    location.wind()['humidity']=location.atmosphere()['humidity']
    location.wind()['pressure']=location.atmosphere()['pressure']
    location.wind()['visibility']=location.atmosphere()['visibility']
    location.wind()['latitude']=location.latitude()
    location.wind()['longitude']=location.longitude()
    #location.wind()['text']=location.condition().text()
    datax.append(location.wind())
 x=pd.DataFrame.from_dict(datax)
```

- The above code retrieves weather data from the yahoo weather website ,using python wrapper (weather-api).
- Weather-api provides access to yahoo weather data , **without any requirement for api-key**.	
- The raw data from yahoo weather website is in json format which can be viewed as given below.

Semi-Structured Raw Data :Json Format


![Image of Plot](https://github.com/SohitKalluru/K-Means-Clustering-of-Major-US-Cities-based-on-Climatic-Conditions-Weather-Data-/blob/master/Images/raw.png)

---
- In the above code the raw data as seen in figure is filtered, transformed:scaled in order to create a good machine learning algorithm
  and restructured according to the requirements.The temperatue,pressure,humidity,visibilty, climatic condition are appended to a       list.This list is later formatted to a dataframe to perform further operations.

Structured Data in the form of Pandas Dataframe:

![Image of Plot](https://github.com/SohitKalluru/K-Means-Clustering-of-Major-US-Cities-based-on-Climatic-Conditions-Weather-Data-/blob/master/Images/df.png)

```
from sklearn.preprocessing import MinMaxScaler    
x=pd.DataFrame.from_dict(datax)
scaler=MinMaxScaler()
xscaled=pd.DataFrame(scaler.fit_transform(x),columns=x.columns)
x.latitude = x.latitude.astype(float)
x.longitude = x.longitude.astype(float)
xscaled.plot.box()
plt.show()
```

Box Plot
 
![Image of Plot](https://github.com/SohitKalluru/K-Means-Clustering-of-Major-US-Cities-based-on-Climatic-Conditions-Weather-Data-/blob/master/Images/Imagesxscaled.PNG)

- we need to eliminate outliers : Virtual appearence of outliers after scaling.
- For more information regarading outliers - check source.
- Reason : A more robust machine learning algorithm  should be made,free from noise.

```
xscaled.plot.density()
plt.show()

```
Density Plot

![Image of Plot](https://github.com/SohitKalluru/K-Means-Clustering-of-Major-US-Cities-based-on-Climatic-Conditions-Weather-Data-/blob/master/Images/xscaled1.PNG)

- Even after scaling all the variables are not distributed normally: because they data values may be sparsely distributed.
 So, need of principal components: Reducing attributes and capturing the same variance from the original data.


```
#PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
PC=pca.fit_transform(xscaled)
pdf=pd.DataFrame(data=PC,columns=['A','B'])
plt.scatter(pdf['A'],pdf['B'])

```
PCA Plot:

![Image of Plot](https://github.com/SohitKalluru/K-Means-Clustering-of-Major-US-Cities-based-on-Climatic-Conditions-Weather-Data-/blob/master/Images/pca.PNG)

- After performing PCA, checking whether the co-relation exists using the graph.In the above case we can infer that we have uncorrelated Principle components.

```
# PCA for outlier detection
## We overcame the problem of outliers using PCA
pdf.plot.box()
plt.show()
```
Box Plot after PCA:

![Image of Plot](https://github.com/SohitKalluru/K-Means-Clustering-of-Major-US-Cities-based-on-Climatic-Conditions-Weather-Data-/blob/master/Images/pca_box.PNG)

- Outlier problem is solved by PCA.

```
pdf.plot.density()
plt.show()
```
Density Plot after PCA:

![Image of Plot](https://github.com/SohitKalluru/K-Means-Clustering-of-Major-US-Cities-based-on-Climatic-Conditions-Weather-Data-/blob/master/Images/density.PNG)

- Both Closely Follow normal distruibution
- PCA solves normal distribution problem as well

```
#Kmeans
from sklearn.cluster import KMeans
km = KMeans(n_clusters = 3).fit(pdf)
y_kmeans=km.predict(pdf)
plt.scatter(pdf['A'], pdf['B'], c=y_kmeans, s=50)
centers = km.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, alpha=1);
plt.show()
```

**Output of Kmeans**


![Image of Plot](https://github.com/SohitKalluru/K-Means-Clustering-of-Major-US-Cities-based-on-Climatic-Conditions-Weather-Data-/blob/master/Images/kmeans.PNG)


- The above code performs Standard normalization of data.It converts every column vector to the range, often between zero and one, or so that  the maximum absolute value of each feature is scaled to unit size.

- The Prinicpal Component analysis is utilized for eliminating dimensions.As all the data available is in different dimensions,PCA can  be utilized to eliminate dimensions- number of components is given to be 2(dimensions).

```
labels=pd.DataFrame.from_dict(y_kmeans,dtype='float')
x=x.join(labels)
x=x.rename(columns={0:'labels'})
x=pd.DataFrame.from_dict(x)
locations = x[['latitude', 'longitude']]
locationlist = locations.values.tolist()

```
- Above code joins the input dataframe and the output(labels after kmeans).
- A list for latitude and longitude is created for running a loop.

```
def regioncolors(counter):
    if counter['labels'] == 1:
        return 'green'
    elif counter['labels'] == 0:
        return 'blue'
    elif counter['labels'] == 2:
        return 'red'
    else:
        return 'darkblue'
```
- Function for identifiying each city with respect to its clusters based on the colors.

```
x["color"] = x.apply(regioncolors, axis=1)
```

![Image of Plot](https://github.com/SohitKalluru/K-Means-Clustering-of-Major-US-Cities-based-on-Climatic-Conditions-Weather-Data-/blob/master/Images/filtereddata.PNG)


- The above code is used to apply the function ,which is later appended to the dataframe.



Folium code :

- The type of map used here is openstreetmap from Folium. There are lot of maps which can be used based on the requirement
- Initial step is setting the home position for the map. 

```
map3 = folium.Map(location=[34.21,-77.88], tiles='CartoDB positron', zoom_start=11)

for point in range(0, len(locationlist)):
    folium.Marker(locationlist[point],popup=CityName[point], icon=folium.Icon(color=x["color"][point], icon_color='white', icon='star', angle=0, prefix='fa')).add_to(map3)

display(map)
```

**CLustered Cities on Map (by Colour)*


![Image of Plot](https://github.com/SohitKalluru/K-Means-Clustering-of-Major-US-Cities-based-on-Climatic-Conditions-Weather-Data-/blob/master/Images/final.PNG)

## How to Run the Code

1. Open the downloaded file by saving it in a juypter working directory

2. Use `run` command to execute the python file .

3. Type the following command:
	```
	run weather_data.py
	```

---

## Suggestions

- This data when combined with data of logistics company along with live traffic data can provide a better understanding in transportation time required between different places.It can also help in determining the days which are favourable for transportation vs days unfavourable. 

 
