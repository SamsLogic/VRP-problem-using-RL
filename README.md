# VRP probelm using RL

## Problem statement

Assume that you are made as a Service Manager for a maintenance company. A maintenance company typically consists of Technicians who drive around & attend maintenance jobs and Service Managers who manage & help these technicians to go around in an optimal way, driving as less distance as possible and attending as many maintenance jobs as possible.

In general, maintenance companies have contracts with their customers which states how many times they have to visit and do a regular check. Hence, Service Managers plan technician visits in advance, generally on a monthly basis. The plan includes a schedule (like a time table) indicating which technician goes where, at what time. These maintenance jobs have a specific duration, say for example 60 minutes (the idea behind this is that the Technician needs 60 minutes to do each maintenance job, this doesn't include the driving time).

The plan that the Service Managers create for Technicians holds good as long as Technicians follow it in the same order. The challenge is that Technicians get callouts and emergency calls during the day. These calls are from their customers asking them to come and visit. This could be for various reasons like - AC is not working, people got stuck inside an Elevator, building fire alarms are not working etc. When Technicians go to attend these on demand orders, they deviate from the plan and their route may not be optimal anymore.

Your task is to attempt finding a way that could automate your (Service Manager's) manual work of creating a plan for Technicians. Also, the solution should consider how to handle the scenarios of callouts and emergencies.

The solution that you come up with might not handle all the use cases. The problem is even broader than what is mentioned above. So it's absolutely fine. What we like to see in the solution is your understanding of the problem and the approach you took to solve it.

## Approach

To solve the above mentioned problem i used Multi Agent Deep Q Reinforcement learning approach in reinforcement learning we follow a feedback based learning where we have an actor that performs an action in a given environment and the feedback(reward) is given as a result of the action along with the observation from the environment. RL follows Markov decision process.

### Assumptions

To solve the above problem we made certain assumption:
1. The service provider have multiple technicians
2. A technicain works only 8 - 10 hours in a day
3. There are 30 days in a month(can be changed for every month)
4. Starting location of the whole procedure is the first cusotmer id that is supposed to be visited (as per the data)
5. the duration and distance matrix is randomly calculated

### Procedure followed

#### Environemnt
The first step for the approach was to build an environment for the given problem.
1. Environment is responsible for state transition to the next location(implemeneted in the env.step function)
2. a step in an environment is completed whenever a location(state) is reached which takes the minimum time to complete.
3. Completion of a state occurs when the duration to reach a state from the previous state + 60 minutes work time is reached.
4. Observations from the environment includes:
   - Next state
   - All states reached or not (boolean)
   - Reward
   - Work completed at a location(state) in a single step
5. Whenever all the states are reached or 30 days are reached in a procedure the environment resets and the whole procedure starts again from the first location.

#### Actor
There can be n number of actors in the environment. These actors each perform certain action which leads to state transition(going to another cusotmer). An actor perfroms one action which identifies the location that the actor is supposed to go next.

#### Policy
An "Off policy" method is followed in the approach where we are using a deep learning model to predict the action to be performed with no specific critic or policy defining the action to be used initially.

To explore multiple possibilties of determining the next states an epsilon-greedy approach was followed.

A deterministic approach (actor critic) is followed in the second code but it is still under development(VRP problem using RL model2.py).

#### Input 

Inputs for the code include:
1. Number of technicians
2. duration matrix
3. distance matrix
4. Number of days
*Please Note : The above mentioned inputs are hardcoded in the code functionality for inputs can be added in the command line later on*

#### Output

The output of the above mentioned code would be a timesheet for all the technicians for a period of 30 days.
The timesheet includes daily schedule of a technician to reachout to the specified customers to cover every customer within 30 days with minimum time consumed with those 30 days.

## How to run the code

### Required packages

1. Pandas (Version = 0.25.2)
2. Numpy (Version = 1.19.5)
3. Polyline (Version = 1.4.0)
4. Tensorflow (Version = 2.5.0)
5. Tensorflow GPU (Version = 2.5.0)

You can directly install the above mentioned packages by using the command:

`pip install -r requirements.txt --user`

### Train

Clone the repository and type the below mentioned command to train the RL model:

`git clone https://github.com/SamsLogic/VRP-probelm-using-RL`

Change directory to the cloned directory

`python VRP_RL_model.py`

Note: Make sure the files are placed as per the repository.

### Using the Model trained

Use the model generated above the create the required timesheet

`python Timesheet_generator.py`

*Please Note: you can directly run this code as per the model trained before during the testing of the above code. Also in order to run the trained model change the name of the trained model as "model.h5" *

## Limitations

There are some limitations in the above code that still requires attention. For now i believe the code mentioned is good for a basic start in solving the complex problem mentioned above. The limitations are:

1. Emergency condition is not yet developed (can be easilty added in the future)
2. The reward function can be imporved as the current reward function may not provide the most optimal solution as per the problem
3. Minimum technicains required to complete the task within the days provided (30 in our case) should be calculated before hand (Can be calculated in the future in realtime only)
4. The timesheet provided can be imporved as date and time can also be incorporated in the timesheet.

