import pandas as pd
import numpy as np
import polyline

from tqdm.notebook import tqdm

path_to_file = 'Mainteny_Sample_Data_Elevators.xlsx'
data = pd.read_excel(path_to_file)

data = data.sort_values(by=['Last visit'])

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

# Number of Technicians in the company
technicians = 5

# Days in a month
work_days_per_month = 30

# Per day 8 hours of work
work_minutes_per_day = 8*60

print(distance_mat)
print(duration_mat)

# Episode Length is a month
episode_length = work_days_per_month


class VRP_problem:
    def __init__(self,action_spec,obersvation_spec,duration_mat,distance_mat,days,minutes_per_day,technicians):
        self._action_spec = action_spec
        self._observation_spec = observation_spec
        self._state = np.zeros(technicians,dtype=np.int32)
        self._episode_end = 0
        self._reward = 0
        self._duration_mat = duration_mat
        self._distance_mat = distance_mat
        self._states_travelled = {x:[0] for x in range(technicians)}
        self._days = days
        self._minutes_per_day = minutes_per_day
        self._technicains = technicians
        
    def reward(self):
        return self._reward
    
    def action_spec(self):
        return self._action_spec
        
    def observation_spec(self):
        return self._observation_spec
        
    def states_tarvelled(self):
        return self._states_travelled
        
    def reset(self):
        self._state = []
        self._episode_end = 1
        self.reward = 0
        
    def observation(self):
        minutes_left_in_day = self._minutes_per_day - 
        
        
    def step(self, action):
        if self._episode_end == 1:
            self.reset()
            return _
        
        for i in range(self._technicains):
            self._state[i] = self._customer_id[self._state[i]][action[i]]
            self._states_travelled[i].append(self._state[i])
        
        self._reward = self._reward + 
        
        
        