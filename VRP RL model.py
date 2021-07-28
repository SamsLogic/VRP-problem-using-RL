import pandas as pd
import numpy as np
import polyline
import random
from tqdm.notebook import tqdm

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

# Number of Technicians in the company
technicians = 20

# Days in a month
work_days_per_month = 30

# Per day 8 hours of work
work_minutes_per_day = 8*60

# Episode Length is a month
episode_length = work_days_per_month

# Configuration parameter
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (epsilon_max - epsilon_min)

batch_size = 16
max_steps_per_episode = 10000

class VRP_problem:
    def __init__(self,cusomter_list,duration_mat,distance_mat,days,minutes_per_day,technicians):
        self._state = np.zeros(technicians,dtype=np.int32)
        self._episode_end = 0
        self._reward = 0
        self._customer_id = cusomter_list
        self._duration_mat = duration_mat
        self._distance_mat = distance_mat
        self._states_travelled = {x:[0] for x in range(technicians)}
        self._days = days
        self._minutes_per_day = minutes_per_day
        self._technicians = technicians
        self._activity_time = np.zeros(technicians,dtype=np.float32)
        self._time_in_a_day = np.array([minutes_per_day]*(technicians),dtype=np.float32) 
        self._maximum_actions = 2776
        
    def reward(self):
        return self._reward
        
    def states_tarvelled(self):
        return self._states_travelled
        
    def reset(self):
        self._state = np.zeros(self._technicians,dtype=np.int32)
        self._episode_end = 0
        self.reward = 0
        self._maximum_actions = 2776
        self._states_travelled = {x:[0] for x in range(self._technicians)}
        self._activity_time = np.zeros(self._technicians,dtype=np.float32)
        self._time_in_a_day = np.array([self._minutes_per_day]*(self._technicians),dtype=np.float32) 
        
    def reset_time(self):
        self._time_in_a_day = np.array([self._minutes_per_day]*(self._technicians),dtype=np.float32) 
        
    # Every step is of 60 mins
    def step(self, action,emergency=0):
        if self._episode_end == 1:
            self.reset()
            return self._state,self._reward, self._episode_end, self._states_travelled
        
        # Action is a dictionary of technician and customer to transit to
        for technician, customer_id in action.items():
            # Activity time calculation
            self._activity_time[technician] = self._duration_mat[self._state[technician]][customer_id]
            # State transition
            self._state[technician] = self._customer_id[customer_id]
            # State hisotry storage
            self._states_travelled[technician].append(self._state[technician])
            
        try:  
            min_time = np.min(self._activity_time[np.nonzero(self._activity_time)])
        except:
            min_time = 0
        
        # Calculating time left in a day
        self._time_in_a_day = self._time_in_a_day - min_time
        
        # Actions completed 
        action_comp = (self._activity_time == min_time)*1
        
        # Activity time updation
        self._activity_time = self._activity_time - min_time
        
        self._maximum_actions -= sum(action_comp)
        # Reward - +100 for every service completed and +200 for every emergency handled        
        self._reward = self._reward + 100*(sum(action_comp)) + 200*(emergency)
        
        if min_time == 0:
            self._episode_end = 1
            print('min')
            return self._state,self._reward, self._episode_end, self._states_travelled
        
        if sum(self._time_in_a_day) <=0:
            self.reset_time()        
            self._activity_time = self._activity_time + (self._activity_time != min_time)*min_time
            self._days -= 1
        
        if self._days == 0:
            self._episode_end = 1
            print('days')
            return self._state,self._reward, self._episode_end, self._states_travelled
            
        elif self._maximum_actions <=0:
            print('max')
            self._episode_end = 1
            return self._state,self._reward, self._episode_end, self._states_travelled
            
        return self._state,self._reward, self._episode_end, self._states_travelled
    
env = VRP_problem(data['Customer ID'].index,duration_mat,distance_mat,work_days_per_month,work_minutes_per_day,technicians)

def simulate_env(env,technicians):
    env.reset()
    state_space = list(range(1,2777))
    random.shuffle(state_space)
    for step in range(max_steps_per_episode):
        action = {x:state_space.pop() for x in range(technicians)}
        state, reward, done, history = env.step(action)
        print(f"Step {step+1}\n")
        print("State: " , state)
        print("reward: ", reward)
        print("history: ",history)
        print("Done: ", done)
        if done == 1 or len(state_space) < technicians :
            print(max([max(x) for _,x in history.items()]))
            break

simulate_env(env,technicians)