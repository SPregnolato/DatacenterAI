
#TRAINING THE AI

#Installing Keras
# conda install -c conda-forge keras

#Libraries
import os
import numpy as np
import random as rn
import DQL_environment
import DQL_brain
import DQL_dqn

#Setting seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

#Setting Parameters
epsilon = .3
number_actions = 5
direction_boundary = (number_actions - 1) / 2
number_epochs = 100
max_memory = 3000
batch_size = 512
temperature_step = 1.5

#Building the environment
env = DQL_environment.Environment(optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)

#Building the brain
brain = DQL_brain.Brain(learning_rate = 0.00001, number_actions = number_actions)

#Building the model
dqn = DQL_dqn.DQN(max_memory = max_memory, discount = 0.9)


#Training Mode = on
train = True

#Training the AI
env.train = train
model = brain.model
early_stopping = True
patience = 10
best_total_reward = -np.inf
patience_count = 0

if (env.train):
    # Loop over Epochs (1 Epoch = 5 Months)
    for epoch in range(1, number_epochs):
        total_reward = 0
        loss = 0.
        new_month = np.random.randint(0, 12)
        env.reset(new_month = new_month)
        game_over = False
        current_state, _, _ = env.observe()
        timestep = 0
        #Loop over Timesteps (1 Timestep = 1 Minute) in one Epoch
        while ((not game_over) and timestep <= 5 * 30 * 24 * 60):
            #Play action by Exploration
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, number_actions)
                if (action - direction_boundary < 0):
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temperature_step
            #Play action by Inference
            else:
                q_values = model.predict(current_state)
                action = np.argmax(q_values[0])
                if (action - direction_boundary < 0):
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temperature_step
                
            #Environment update: next state
            next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep / (30*24*60)))
            total_reward += reward
            
            #Storing Transition in Memory
            dqn.remember([current_state, action, reward, next_state], game_over)
            
            #Gathering Inputs and Targets in separate Batches
            inputs, targets = dqn.get_batch(model, batch_size = batch_size)
            
            #Compute the loss over the all Batches 
            loss += model.train_on_batch(inputs, targets)
            timestep += 1
            current_state = next_state
    
        #Printing training result for each Epoch
        print("\n")
        print("Epoch: {:03d}/{:03d}".format(epoch, number_epochs))
        print("Total Energy spent with an AI: {:.0f}".format(env.total_energy_ai))
        print("Total Energy spent with no AI: {:.0f}".format(env.total_energy_noai))
        
        #Early stopping
        if (early_stopping):
            if (total_reward <= best_total_reward):
                patience_count += 1
            elif (total_reward > best_total_reward):
                best_total_reward = total_reward
                patience_count = 0
            
            if (patience_count >= patience):
                print("Early Stopping")
                break

        #Saving the model
        model.save("model.h5")
        





