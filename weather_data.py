import folium
from weather import Weather, Unit
import pandas as pd
import matplotlib.pyplot as plt
weather = Weather(unit=Unit.CELSIUS)
from weather import Weather, Unit

places=[2459115,2442047,2379574,2424766,2471390,2471217,2487796,2388929,2487889,2488042,2391585,2487956,2428344,2427032,2357536,2383660,28340407,2378426,2449323,2358820,2367105,2397816,2451822,2391279,2490383,2457170,2514815,2436704,2475687,2442327,2464592,2508428,2357024,2352824,2430683,2407517,2486340,2441472,2449808,2465512,2381475,2512636,2450022,2463583,2478307,2508533,2452078,2383489,2423945,2355944,2520077,2486982,2503863,2488802,2458833,2354447,2380358,2358492,2357473,2473224,2482128,2506911,2500105,2385304,2438841,2487129,2354490,2459269,2371464,2473475,2419946,2439482,2406008,2411084,2414469,2378015,2487180,2429187,2490057,2460389,2443945,2466256,2364559,2359991,2394734,2436565,2442818,2379200,2380213,2408976,2522292,2461253,2480201,2410128,2420610,2355942,2352491,2427665,2482949,2366355,2452629,2407405,2453369,2497646,2480894,2524811,2427690,2493227,2487870,2503523,2411009,2391446,2357383,2412843,2425873,2452537,2453984,2440351,2354141,2383661,2467212,2404850,2433662,2405797,2523945,2487610,2459618,2426010,2504633,2370568,2402726,2428184,2503713,2357467,2465715,2477058,2466942,2478522,2378695,2464118,2488845,2408784,2511258,2412837,2470457,2483357,2374635,2498315,2488916,2494126,2475492,2389876,2487384,2474876,2498304,2400539,2385250,2468963,2429708,2470103,2469081,2416847,2436084,2353019,2487460,2467721,2457000,2468964,2430632,2419175,2423467,2435724,2507261,2503418,2400183,2405641,2368947,2465890,2514383,2398401,2489314,2449851,2502265,2408095,2447466,2375810,2376926,2499659,2383552,2384895,2375543,2398316,2418244,2512937,2362031,2458410,2517863,2507158,2505987,2397796,2426709,2448187,2384020,2512682,2493889,2464639,2380893,2391230,2498846,2477080,2498306,2432286,2351598,2400767,2408354,2510744,2354842,2470456,2436453,2434560,2505922,2356940,2404367,2427199,2484861,2378319,2360899,2512106,2488836,2385447,2450083,2444674,2452272,2393444,2356381,2353412,2518344,2515048,2460448,2450465,2398255,2516864,2381303,2373572,2477147,2517245,2485177,2364254,2400052,2495968,23417225,2401427,2442564,2462248,2371863,2480904,2474897,2421250,2456416,2438795,2480733,2389087,2362930,2414913,2520100,2413753,2389559,2467662,2383559,2475747,2482950,2355124,2521361]
# Lookup via location name.
city = pd.DataFrame.from_dict(places)
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

from sklearn.preprocessing import MinMaxScaler    
x=pd.DataFrame.from_dict(datax)
scaler=MinMaxScaler()
xx=x.drop(['longitude','latitude'],axis=1)
xscaled=pd.DataFrame(scaler.fit_transform(xx),columns=xx.columns)
x.latitude = x.latitude.astype(float)
x.longitude = x.longitude.astype(float)


xscaled.plot.box()
plt.show()

xscaled.plot.density()
plt.show()

#PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
PC=pca.fit_transform(xscaled)
pdf=pd.DataFrame(data=PC,columns=['A','B'])
plt.scatter(pdf['A'],pdf['B'])
plt.show()

pdf.plot.box()
plt.show()

pdf.plot.density()
plt.show()

# Kmeans
from sklearn.cluster import KMeans
km = KMeans(n_clusters = 3).fit(pdf)
y_kmeans=km.predict(pdf)
plt.scatter(pdf['A'], pdf['B'], c=y_kmeans, s=50)
centers = km.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, alpha=1);
plt.show()

x=pd.DataFrame.from_dict(datax,dtype='float')
labels=pd.DataFrame.from_dict(y_kmeans,dtype='float')
x.dtypes

x=x.join(labels)
x=x.rename(columns={0:'labels'})
x=pd.DataFrame.from_dict(x)

locations = x[['latitude', 'longitude']]
locationlist = locations.values.tolist()

def regioncolors(counter):
    if counter['labels'] == 1:
        return 'green'
    elif counter['labels'] == 0:
        return 'blue'
    elif counter['labels'] == 2:
        return 'red'
    else:
        return 'darkblue'
x["color"] = x.apply(regioncolors, axis=1)

CityName=["New York","Los Angeles","Chicago","Houston","Phoenix","Philadelphia","San Antonio","Dallas","San Diego","San Jose","Detroit","San Francisco","Jacksonville","Indianapolis","Austin","Columbus","Fort Worth","Charlotte","Memphis","Baltimore","Boston","El Paso","Milwaukee","Denver","Seattle","Nashville","Washington","Las Vegas","Portland","Louisville","Oklahoma City","Tucson","Atlanta","Albuquerque","Kansas City","Fresno","Sacramento","Long Beach","Mesa","Omaha","Cleveland","Virginia Beach","Miami","Oakland","Raleigh","Tulsa","Minneapolis","Colorado Springs","Honolulu","Arlington","Wichita","St. Louis","Tampa","Santa Ana","New Orleans","Anaheim","Cincinnati","Bakersfield","Aurora","Pittsburgh","Riverside","Toledo","Stockton","Corpus Christi","Lexington","St. Paul","Anchorage","Newark","Buffalo","Plano","Henderson","Lincoln","Fort Wayne","Glendale","Greensboro","Chandler","St. Petersburg","Jersey City","Scottsdale","Norfolk","Madison","Orlando","Birmingham","Baton Rouge","Durham","Laredo","Lubbock","Chesapeake","Chula Vista","Garland","Winston-Salem","North Las Vegas","Reno","Gilbert","Hialeah","Arlington","Akron","Irvine","Rochester","Boise","Modesto","Fremont","Montgomery","Spokane","Richmond","Yonkers","Irving","Shreveport","San Bernardino","Tacoma","Glendale","Des Moines","Augusta","Grand Rapids","Huntington Beach","Mobile","Moreno Valley","Little Rock","Amarillo","Columbus","Oxnard","Fontana","Knoxville","Fort Lauderdale","Worcester","Salt Lake City","Newport News","Huntsville","Tempe","Brownsville","Fayetteville","Jackson","Tallahassee","Aurora","Ontario","Providence","Overland Park","Rancho Cucamonga","Chattanooga","Oceanside","Santa Clarita","Garden Grove","Vancouver","Grand Prairie","Peoria","Rockford","Cape Coral","Springfield","Santa Rosa","Sioux Falls","Port St. Lucie","Dayton","Salem","Pomona","Springfield","Eugene","Corona","Pasadena","Joliet","Pembroke Pines","Paterson","Hampton","Lancaster","Alexandria","Salinas","Palmdale","Naperville","Pasadena","Kansas City","Hayward","Hollywood","Lakewood","Torrance","Syracuse","Escondido","Fort Collins","Bridgeport","Orange","Warren","Elk Grove","Savannah","Mesquite","Sunnyvale","Fullerton","McAllen","Cary","Cedar Rapids","Sterling Heights","Columbia","Coral Springs","Carrollton","Elizabeth","Hartford","Waco","Bellevue","New Haven","West Valley City","Topeka","Thousand Oaks","El Monte","Independence","McKinney","Concord","Visalia","Simi Valley","Olathe","Clarksville","Denton","Stamford","Provo","Springfield","Killeen","Abilene","Evansville","Gainesville","Vallejo","Ann Arbor","Peoria","Lansing","Lafayette","Thornton","Athens","Flint","Inglewood","Roseville","Charleston","Beaumont","Victorville","Santa Clara","Costa Mesa","Miami Gardens","Manchester","Miramar","Downey","Arvada","Allentown","Westminster","Waterbury","Norman","Midland","Elgin","West Covina","Clearwater","Cambridge","Pueblo","West Jordan","Round Rock","Billings","Erie","South Bend","San Buenaventura (Ventura)","Fairfield","Lowell","Norwalk","Burbank","Richmond","Pompano Beach","High Point","Murfreesboro","Lewisville","Richardson","Daly City","Berkeley","Gresham","Wichita Falls","Green Bay","Davenport","Palm Bay","Columbia","Portsmouth","Rochester","Antioch","Wilmington"]
map3 = folium.Map(location=[34.21,-77.88], tiles='CartoDB positron', zoom_start=11)
for point in range(0, len(locationlist)):
    folium.Marker(locationlist[point],popup=CityName[point], icon=folium.Icon(color=x["color"][point], icon_color='white', icon='star', angle=0, prefix='fa')).add_to(map3)
display(map3)


