from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam



import numpy as np
from collections import deque
import scipy.special as ssp

# POLICY --> TAKE ACTION
def policy(model, current_state, tau_soft, number_actions, direction_boundary, temperature_step):
    
    # Numpy to Torch
    current_state_t = torch.from_numpy(current_state).float()
    # prediction
    q_values_t = model(current_state_t) 
    #Torch to numpy
    q_values = q_values_t.detach().numpy().squeeze()
    
    # normalization
    q_values_norm = (q_values-min(q_values)) / (max(q_values)-min(q_values))
    # softmax
    probs = ssp.softmax(q_values_norm/tau_soft - max(q_values_norm/tau_soft))
    action = np.random.choice(number_actions, p = probs)
    # q_hat for avg reward update
    q_hat = q_values[action]
    # action to energy
    if (action - direction_boundary < 0):
        direction = -1
    else:
        direction = 1
    energy_ai = abs(action - direction_boundary) * temperature_step

    return action, q_hat, direction, energy_ai


# BUILDING THE BRAIN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random

# define the network
class Net(nn.Module):

    def __init__(self, n_actions = 5, n_states = 3):
        super(Net, self).__init__()
        self.n_actions = n_actions
        self.n_states = n_states
        # z[l] = w * a[l-1] + b
        self.fc1 = nn.Linear(n_states, 64)  
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_actions)

    def forward(self, state):
        # a[l] = g(z[l])
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        q_values = self.fc3(a)
        return q_values



# DEEP Q-LEARNING ALGORITHM (With Experience Replay)
class DQN():
    
    def __init__(self, max_memory = 10, discount = 0.99, n_actions = 5, n_states = 3):
        self.memory = deque(maxlen=max_memory)
        self.max_memory = max_memory
        self.discount = discount
        self.model = Net(n_actions, n_states)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
    
    # Methods to build the memory in Experience Replay
    def remember(self, transition):
        self.memory.append(transition)
        
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):      
        # zero the parameter gradients
        self.optimizer.zero_grad()
        # forward + backward + optimize
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        criterion =  F.smooth_l1_loss()
        td_loss = criterion(outputs, target)
        td_loss.backward(retain_graph = True)
        self.optimizer.step()
    
    def update(self, reward, current_state, next_state, action, batch_size):
        current_state = torch.Tensor(current_state).float().unsqueeze(0)  
        new_state = torch.Tensor(next_state).float().unsqueeze(0)
        self.remember((current_state, new_state, torch.LongTensor([int(action)]), torch.Tensor([reward])))
        actual_batch_size = min(len(self.memory), batch_size)      
        batch_state, batch_next_state, batch_action, batch_reward = self.sample(actual_batch_size)
        self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        



# BUILDING THE BRAIN
class Brain():
    
    def __init__ (self, learning_rate = 0.001, number_actions = 5, number_states = 3, loss = 'mean_squared_error', optimizer = Adam()):
        
        self.learning_rate = learning_rate 
        
        # Input layer
        states = Input(shape = (number_states,))
        
        # Hidden layers
        y = Dense(units = 64, activation = 'relu', kernel_initializer='he_normal', bias_initializer='zeros')(states)
        y = Dropout(rate = 0.2)(y)
     
        y = Dense(units = 32, activation = 'relu', kernel_initializer='he_normal', bias_initializer='zeros')(y)
        y = Dropout(rate = 0.1)(y)
        
        # y = Dense(units = 16, activation = 'relu', kernel_initializer='he_normal', bias_initializer='zeros')(y)
        # y = Dropout(rate = 0.1)(y)
        
        # Output layer
        q_values = Dense(units = number_actions, activation = 'linear', kernel_initializer='he_normal', bias_initializer='zeros')(y)
        # q_values = Dense(units = number_actions, activation = 'tanh')(y)
        # q_values = Dense(units = number_actions, activation = 'softmax')(y)

        # Assembling the full architecture in a model object (object variable)
        self.model = Model(inputs = states, outputs = q_values)
        
        # Compiling the model with loss and optimizer (applying the compile method)
        self.model.compile(loss = loss, optimizer = optimizer)
