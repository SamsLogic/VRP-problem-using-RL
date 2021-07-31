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

batch_size = 32
max_steps_per_episode = 10000

class VRP_problem:
    def __init__(self,cusomter_list,duration_mat,distance_mat,days,minutes_per_day,technicians):
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
    
    def states_travelled(self):
        return self._states_travelled
    
    # Every step is of 60 mins
    def step(self, action):
        if np.all(self._episode_end == 1):
            return self._state,self._reward, self._episode_end, self._action_comp
        
        # Action is a dictionary of technician and customer to transit to
        for technician, customer_id in action.items():
            # Activity time calculation
            if customer_id == 0:
                break
            self._activity_time[technician] = self._duration_mat[self._state[technician]][customer_id]
            # State transition
            self._state[technician] = self._customer_id[customer_id]
            self._states_travelled[technician].append(self._state[technician])
            
        try:  
            min_time = np.min(self._activity_time[np.nonzero(self._activity_time)])
        except:
            min_time = 0
            self._episode_end = np.ones(self._technicians,dtype=np.int32)
            return self._state,self._reward, self._episode_end, self._action_comp
            
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
        
        temp_reward = (np.zeros(self._technicians,dtype=np.int32)+(min_time)*(self._action_comp))/(self._days+1)
        self._reward -= temp_reward
        # if min_time == 0:
            # self._episode_end = np.ones(self._technicians,dtype=np.int32)
            # print('min')
            # return self._state,self._reward, self._episode_end, self._action_comp       
        
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

# my policy network would a Deep Q network with RNN layers to keep a track of previous states_tarvelled

# When to train the network
update_after_steps = 5

# When to update the traget network
update_target_network = 500

# Number of state to take random action and observe output
epsilon_random_steps = 1000
# Number of states for exploration
epsilon_greedy_steps = 100000

# size of replay buffer
replay_buffer_size = 10000

state_history = []
state_next_history = []
done_history = []
action_history = []
rewards_history = []
episode_reward_history = []
running_reward = np.zeros(technicians,dtype=np.float32)
episode_count = 0
step_count = 0

def create_model(technicians):
    inp = L.Input(shape=(technicians,1))
    x = L.Dense(32,activation = 'relu')(inp)
    x = L.Dense(256,activation='relu')(x)
    x = L.Dense(2777,activation = 'linear')(x)
    return M.Model(inputs=inp,outputs=x)

model = create_model(technicians)
model_target = create_model(technicians)

optimizer = O.Adam(learning_rate=0.01)
loss_fn = Loss.Huber()

# run episodes until solved
while True:
    state = env.reset()
    episode_reward = np.zeros(technicians,dtype=np.float32)
    action_comp = np.ones(technicians,dtype=np.float32)
    state_space = np.arange(0,2777)
    for step in range(1, max_steps_per_episode):
        step_count += 1
        action = np.zeros(technicians,dtype=np.int32)        
        if step_count < epsilon_random_steps or epsilon > np.random.rand(1)[0]:
            for i in np.where(action_comp == 1)[0]:
                try:
                    action[i] = np.random.choice(state_space[state_space>0])
                    state_space[action[i]] = 0
                except:
                    pass
        else:
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
        
        action_history.append(action)
        action = {x:action[x] for x in np.where(action_comp == 1)[0]}
        epsilon -= epsilon_interval/epsilon_greedy_steps
        epsilon = max(epsilon, epsilon_min)
        
        state_next, reward, done, action_comp = env.step(action)

        episode_reward += reward
        
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        
        rewards_history.append(reward)
        state = state_next
        
        if step_count % update_after_steps == 0 and len(done_history) > batch_size:
            indices = np.arange(batch_size)
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = np.array([rewards_history[i] for i in indices])
            action_sample = np.array([action_history[i] for i in indices])
            done_sample = np.array([done_history[i] for i in indices])
            future_rewards = model_target.predict(state_next_sample)
            
            updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards,axis=-1)
            
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample
            print(updated_q_values)
            masks = tf.one_hot(action_sample,2777)
            
            with tf.GradientTape() as tape:
                q_values = model(state_sample)
                q_action = tf.reduce_sum(tf.multiply(q_values, masks),axis=-1)
                loss = loss_fn(updated_q_values,q_action)
                
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))
        
        if step_count % update_target_network == 0:
            model_target.set_weights(model.get_weights())
            # print(f"episode reward: {episode_reward} at episode {episode_count}, step count {step_count}")
        
        if len(rewards_history) > replay_buffer_size:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]
        if np.all(done == 1):
            break
    print(len(state_space[state_space>0]))
    print(f"episode reward: {episode_reward} at episode {episode_count}, step count {step_count}")
    episode_reward_history.append(list(episode_reward))
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]

    episode_count += 1
    # print(env.states_travelled())
    # print(sum(len(val) for _,val in env.states_travelled().items()))
    # Condition to consider the task solved
    if np.all(episode_reward > -1000):  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break



