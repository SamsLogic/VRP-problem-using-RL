pip install -U googlemaps polyline

import googlemaps
import pandas as pd
import numpy as np
import polyline

from tqdm.notebook import tqdm

gmaps = googlemaps.Client(key='AIzaSyCbl9uMGsldUysJwAkWnoF3MataSKeuZ10')

path_to_file = 'Mainteny_Sample_Data_Elevators.xlsx'
data = pd.read_excel(path_to_file)

data

data['ZIP Code'] = data['ZIP Code'].astype('str')
data['addr'] = data['City'].str.cat(data['ZIP Code'],sep=', ')
data = data.sort_values(by=['Last visit'])
data = data.drop_duplicates(['addr'])

data

duration = np.zeros((190,190))
distance = np.zeros((190,190))

steps = 10
for k in tqdm(range(len(data['addr'].values))):
    for i in range(0,len(data['addr'].values),steps):
        dist = gmaps.distance_matrix(origins=data['addr'].values[k],destinations=data['addr'].values[i:i+steps],mode='driving')
        for j in range(steps):
            try:
                if i+j == k:
                    continue
                distance[k][i+j] = dist['rows'][0]['elements'][j]['distance']['value']
                duration[k][i+j] = dist['rows'][0]['elements'][j]['duration']['value']
            except:
                break

distance_matrix = pd.DataFrame(distance)
duration_matrix = pd.DataFrame(duration)

distance_matrix.to_csv('distance_matrix.csv',header=None)
duration_matrix.to_csv('duration_matrix.csv',header=None)

distance_matrix = pd.read_csv('distance_matrix.csv',header=None)
duration_matrix = pd.read_csv('duration_matrix.csv',header=None)

