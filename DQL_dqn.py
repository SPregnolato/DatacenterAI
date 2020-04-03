
#DEEP Q-LEARNING ALGORITHM (With Experience Repaly)
import numpy as np
from collections import deque
 
class DQN(object):
    #Initializing parameters and variables
    def __init__(self, max_memory = 100, discount = 0.99):
        self.memory = deque(maxlen=max_memory)
        self.max_memory = max_memory
        self.discount = discount
    
    #Methods to build the memory in Experience Replay
    def remember(self, transition, game_over):
        self.memory.append([transition, game_over])
    # def remember(self, transition, game_over):
    #     self.memory.append([transition, game_over])
    #     if len(self.memory) > self.max_memory:
    #         del self.memory[0]
    
   
    #Methods to build two batches of 10 In and 10 Targes by extracting 10 transition
    # def get_batch(self, model, batch_size):
    def get_batch(self, model, batch_size, r_hat):
        len_memory = len(self.memory)
        num_inputs = self.memory[0][0][0].shape[1]
        num_outputs = model.output_shape[-1]
        inputs = np.zeros((min(len_memory, batch_size), num_inputs))
        targets = np.zeros((min(len_memory, batch_size), num_outputs))
        for i, idx in enumerate(np.random.randint(0, len_memory, size = min(len_memory, batch_size))):
            current_state, action, reward, next_state, r_hat_i = self.memory[idx][0]
            game_over = self.memory[idx][1]
            inputs[i] = current_state
            targets[i] = model.predict(current_state)[0]
            Q_sa_next = np.max(model.predict(next_state)[0])   
            if game_over:
                # targets[i, action] = reward - r_hat
                # targets[i, action] = reward - r_hat_i
                targets[i, action] = reward 
            else:
                # targets[i, action] = reward - r_hat + self.discount * Q_sa_next
                # targets[i, action] = reward - r_hat_i + self.discount * Q_sa_next
                targets[i, action] = reward + self.discount * Q_sa_next
        return inputs, targets

   