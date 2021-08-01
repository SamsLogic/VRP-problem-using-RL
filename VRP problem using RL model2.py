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

# Episode Length is a month
episode_length = work_days_per_month

# Configuration parameter
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (epsilon_max - epsilon_min)

batch_size = 32
tau = 0.005
max_steps_per_episode = 10000

class VRP_problem:
    def __init__(self,cusomter_list,duration_mat,distance_mat,days,minutes_per_day,technicians):
        self._time_consumed = np.zeros(technicians,dtype=np.float32)
        self._state = np.zeros(technicians,dtype=np.int32)
        self._episode_end = np.zeros(technicians,dtype=np.int32)
        self._reward = np.zeros(technicians,dtype=np.float32)
        self._customer_id = cusomter_list
        self._duration_mat = duration_mat
        self._distance_mat = distance_mat
        self._action_comp = np.ones(technicians,dtype=np.float32)
        self._days = days
        self._days_total = days
        self._minutes_per_day = minutes_per_day
        self._technicians = technicians
        self._activity_time = np.zeros(technicians,dtype=np.float32)
        self._time_in_a_day = np.array([minutes_per_day]*(technicians),dtype=np.float32) 
        self._maximum_actions = 2776
        self._states_travelled = {x:[0] for x in range(technicians)}
        
    def reward(self):
        return self._reward
        
    def reset(self):
        self._time_consumed = np.zeros(technicians,dtype=np.float32)
        self._state = np.zeros(self._technicians,dtype=np.int32)
        self._episode_end = np.zeros(self._technicians,dtype=np.int32)
        self._reward = np.zeros(self._technicians,dtype=np.float32)
        self._maximum_actions = 2776
        self._states_travelled = {x:[0] for x in range(technicians)}
        self._days = self._days_total
        self._action_comp = np.ones(self._technicians,dtype=np.float32)
        self._activity_time = np.zeros(self._technicians,dtype=np.float32)
        self._time_in_a_day = np.array([self._minutes_per_day]*(self._technicians),dtype=np.float32) 
        return self._state
        
    def reset_time(self):
        self._time_in_a_day = np.array([self._minutes_per_day]*(self._technicians),dtype=np.float32) 
    
    def time_consumed(self):
        return self._time_consumed
    
    def states_travelled(self):
        return self._states_travelled
    
    # Every step is of 60 mins
    def step(self, action):
        if np.all(self._episode_end == 1):
            return self._state,self._reward, self._episode_end, self._action_comp
        
        # Action is a dictionary of technician and customer to transit to
        for technician, customer_id in enumerate(action):
            # Activity time calculation
            self._activity_time[technician] = self._duration_mat[self._state[technician]][customer_id]
            # State transition
            self._state[technician] = self._customer_id[customer_id]
            self._states_travelled[technician].append(self._state[technician])
        
        self._time_consumed += self._activity_time
        
        try:  
            min_time = np.min(self._activity_time[np.nonzero(self._activity_time)])
        except:
            min_time = 0
            
        # Calculating time left in a day
        self._time_in_a_day = self._time_in_a_day - min_time
        
        if sum(self._time_in_a_day) <0:
            self.reset_time()
            self._days -= 1
            for technician, customer_id in enumerate(self._state):
                # Activity time calculation
                self._activity_time[technician] = self._duration_mat[0][customer_id]

            min_time = np.min(self._activity_time[np.nonzero(self._activity_time)])
        
        self._maximum_actions -= sum(self._action_comp)
        
        # Actions completed 
        self._action_comp = (self._activity_time == min_time)*1
        # Activity time updation
        self._activity_time = self._activity_time - min_time
        
        # temp_reward = (np.zeros(self._technicians,dtype=np.int32)+(min_time)*(self._action_comp))/(self._days+1)
        
        self._reward = -self._time_consumed
        # if min_time == 0:
            # self._episode_end = np.ones(self._technicians,dtype=np.int32)
            # print('min')
            # return self._state,self._reward, self._episode_end, self._action_comp       
        
        if len(action) == len(set(action)):
            self._reward = np.array([-333240]*self._technicians,dtype=np.float32)
            print('same')
            self._episode_end = np.ones(self._technicians,dtype=np.int32)
            return self._state,self._reward, self._episode_end, self._action_comp
            
        if self._days == 0:
            self._reward = np.array([-333240]*self._technicians,dtype=np.float32)
            self._episode_end = np.ones(self._technicians,dtype=np.int32)
            print('days')
            return self._state,self._reward, self._episode_end, self._action_comp
            
        if self._maximum_actions <= 0:
            print('All done')
            self._episode_end = np.ones(self._technicians,dtype=np.int32)
            return self._state,self._reward, self._episode_end, self._action_comp
        
        return self._state,self._reward, self._episode_end, self._action_comp
    
env = VRP_problem(data['Customer ID'].index,duration_mat,distance_mat,work_days_per_month,work_minutes_per_day,technicians)

# Environment simulation to check if the environment is coded properly
def simulate_env(env,technicians):
    env.reset()
    state_space = list(range(1,2777))
    random.shuffle(state_space)
    action_comp = np.ones(technicians,dtype=np.int32)
    for step in range(max_steps_per_episode):
        action = {x:state_space.pop() for x in np.where(action_comp==1)[0]}
        state, reward, done,action_comp = env.step(action)
        print(f"Step {step+1}")
        print("State: " , state)
        print("reward: ", reward)
        print("Done: ", done)
        print("\n")
        if done == 1 or len(state_space) < technicians :
            break

# simulate_env(env,technicians)

# size of replay buffer
replay_buffer_size = 100

state_history = []
state_next_history = []
done_history = []
action_history = []
rewards_history = []
episode_reward_history = []
running_reward = np.zeros(technicians,dtype=np.float32)
episode_count = 0
step_count = 0

def actor_model_create(technicians):
    inp = L.Input(shape=(technicians,1))
    x = L.Dense(256,activation = 'relu')(inp)
    x = L.Dense(256,activation='relu')(x)
    x = L.Dense(1,activation = 'linear')(x)
    
    return M.Model(inputs=inp,outputs=x)

actor_model = actor_model_create(technicians)
actor_model_target = actor_model_create(technicians)

def critic_model_create(technicians):
    inp1 = L.Input(shape=(technicians,1))
    x = L.Dense(16, activation='relu')(inp1)
    x = L.Dense(32, activation='relu')(x)
    
    inp2 = L.Input(shape=(technicians,1))
    x1 = L.Dense(32,activation='relu')(inp2)
    
    concat = L.Concatenate()([x,x1])
    
    out = L.Dense(256,activation='relu')(concat)
    out = L.Dense(256,activation='relu')(out)
    out= L.Dense(1)(out)
    
    model = M.Model([inp1,inp2],out)
    return model

critic_model = critic_model_create(technicians)
critic_model_target = critic_model_create(technicians)

def policy(state):
    state = tf.expand_dims(state,1)
    actions = tf.squeeze(actor_model(state))
    legal_actions = np.clip(actions.numpy(),1,2776)
    legal_actions = legal_actions.astype(np.int32)
    return np.squeeze(legal_actions)

@tf.function
def update(state_batch, action_batch, reward_batch, next_state_batch):
        
        with tf.GradientTape() as tape:
            target_actions = actor_model_target(next_state_batch, training=True)
            y = reward_batch + gamma * tf.squeeze(critic_model_target(
                [next_state_batch, target_actions], training=True
            ))
            critic_value = tf.squeeze(critic_model([state_batch, action_batch], training=True))
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = tf.squeeze(actor_model(state_batch, training=True))
            critic_value = tf.squeeze(critic_model([state_batch, actions], training=True))
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

actor_optimizer = O.Adam(learning_rate=0.1)
critic_optimizer = O.Adam(learning_rate=0.1)
loss_fn = Loss.Huber()

# run episodes until solved
while True:
    state = env.reset()
    episode_reward = np.zeros(technicians,dtype=np.float32)
    action_comp = np.ones(technicians,dtype=np.float32)
    state_space = np.arange(0,2777)
    for step in range(1, max_steps_per_episode):
        step_count += 1
        
        action = policy(state)
                
        state_next, reward, done, action_comp = env.step(action)

        episode_reward += reward
        
        state_history.append(state)
        state_next_history.append(list(state_next))
        done_history.append(done)
        rewards_history.append(reward)
        action_history.append(action)
        state = state_next
        
        if len(done_history) > batch_size:
            indices = np.arange(batch_size)
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = np.array([rewards_history[i] for i in indices])
            action_sample = np.array([action_history[i] for i in indices])
            done_sample = np.array([done_history[i] for i in indices])

            update(state_sample,action_sample,rewards_sample,state_next_sample)
            update_target(actor_model_target.variables,actor_model.variables, tau)
            update_target(critic_model_target.variables,critic_model.variables, tau)
            
        if len(rewards_history) > replay_buffer_size:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]
        if np.all(done == 1):
            break

    print(f"episode reward: {episode_reward} at episode {episode_count}, step count {step_count}, time_consumed {max(env.time_consumed())}")
    episode_reward_history.append(list(episode_reward))
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]

    episode_count += 1
    # print(env.states_travelled())
    # print(sum(len(val) for _,val in env.states_travelled().items()))
    # Condition to consider the task solved
    if np.all(episode_reward > 0):  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break



