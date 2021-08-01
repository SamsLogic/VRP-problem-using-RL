from VRP_RL_model import VRP_problem

import pandas as pd
import numpy as np
import polyline
import random
from tqdm.notebook import tqdm

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras.optimizers as O
import tensorflow.keras.losses as Loss

path_to_file = 'Mainteny_Sample_Data_Elevators.xlsx'
data = pd.read_excel(path_to_file)

data = data.sort_values(by=['Customer ID'])

np.random.seed(10)
duration_mat = np.random.uniform(high=60,low=0,size=(2777,2777))
np.random.seed(10)
distance_mat = np.random.uniform(high=50,low=0,size=(2777,2777))

# Adding the 60 min time of service work.
duration_mat = duration_mat+60

# Setting the distance between same locations to 0
for i in range(len(duration_mat)):
    duration_mat[i][i] = 0
    distance_mat[i][i] = 0

duration_mat = duration_mat.astype(np.int32)
distance_mat = distance_mat.astype(np.int32)

# Number of Technicians in the company
technicians = 15

# Days in a month
work_days_per_month = 30

# Per day 8 hours of work
work_minutes_per_day = 10*60

env = VRP_problem(data['Customer ID'].index,duration_mat,distance_mat,work_days_per_month,work_minutes_per_day,technicians)

model = M.load_model('model.h5')

def get_data(states_travelled):
    columns = ['Days']
    data = pd.DataFrame(columns = columns)
    data['Days'] = list(range(30))
    d = np.empty(shape=(30,technicians),dtype=np.object)
    for i in range(len(states_travelled)):
        for j in data['Days'].values:
            stri = ', '.join(str(x) for x in states_travelled[i][j])
            d[j][i] = stri
        data[f'Technician {i+1}'] = d[:,i]
    return data

# run episodes until solved
state = env.reset()
action_comp = np.ones(technicians,dtype=np.float32)
state_space = np.arange(0,2777)

while True:
    action = np.zeros(technicians,dtype=np.int32)
    
    state_ten = tf.convert_to_tensor(state)
    state_ten = tf.expand_dims(state_ten,1)
    state_ten = tf.expand_dims(state_ten,0)
    action_probs = model(state_ten,training=False)
    action_probs = action_probs[0]
    for i in np.where(action_comp ==1)[0]:
        temp_state_space = state_space.copy()
        temp_state_space[temp_state_space > 0]  = 1
        action_probs = action_probs*temp_state_space
        action[i] = tf.argmax(action_probs[i],axis=-1).numpy()
        state_space[action[i]] = 0

    action = {x:action[x] for x in np.where(action_comp == 1)[0]}
    
    state_next, reward, done, action_comp = env.step(action)

    state = state_next
    if np.all(done == 1):
        break

timesheet = get_data(env.states_travelled())
timesheet.to_csv('Timesheet.csv',index=False)
print('Timesheet Generated')
