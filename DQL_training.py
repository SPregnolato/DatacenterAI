
#TRAINING THE AI

#Installing Keras
# conda install -c conda-forge keras

#Libraries

import numpy as np
import DQL_environment
import DQL_brain
import DQL_dqn
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import keras.backend as K
import pickle

#Setting Parameters
number_epochs = 1000
epoch_len = 2 * 30 * 24 * 60   
learning_rate = 0.01
loss_f = 'mean_squared_logarithmic_error'  
decay = 1e-2 / number_epochs
opt = Adam(learning_rate=learning_rate, decay=decay, beta_1=0.95, beta_2=0.999, amsgrad=False)
max_memory = 30 * 24 * 60  
batch_size = 300

r_hat = 0 # !!!!! not used in this version !!!!!
beta = 0.01 # avg reward step
discount = 0.99 # discount factor

number_actions = 7
direction_boundary = (number_actions - 1) / 2
temperature_step = 1.5
max_energy = direction_boundary * temperature_step
optimal_temperature = (20.0, 24.0)

#Building the environment
env = DQL_environment.Environment(optimal_temperature = optimal_temperature, initial_month = 0, initial_number_users = 20, initial_rate_data = 30, max_energy = max_energy)
current_state, _, _ = env.observe()
number_states = current_state.shape[1]

#Building the brain
brain = DQL_brain.Brain(learning_rate, number_actions, number_states,  loss_f, opt)

#Building the model
dqn = DQL_dqn.DQN(max_memory = max_memory, discount = discount)

#Training the AI
train = True
env.train = train
model = brain.model

rew_plot = []
AVG_rew_plot = []
AVG_rew_plot_2 = []
losses_plot = []
AVG_losses_plot = []
r_hat_plot = []
plt.figure()

if (env.train):
    # Loop over Epochs (1 Epoch = 2 Months)
    for epoch in range(1, number_epochs+1):
        total_reward = 0
        loss = 0.
        new_month = np.random.randint(0, 12-int(epoch_len / (30*24*60)))
        env.reset(new_month = new_month)
        game_over = False
        current_state, _, _ = env.observe()
        timestep = 0
        t_in_ai = 0
        t_in_noai = 0
        mse_T_ai = 0
        mse_T_noai = 0
    
        #Loop over Timesteps (1 Timestep = 1 Minute) in one Epoch
        for timestep in range(epoch_len):
            if not game_over: 
                
                # select action a ( softmax policy)
                q_values = model.predict(current_state)[0]
                action = np.random.choice(number_actions, p = q_values )     
                if (action - direction_boundary < 0):
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temperature_step
                    
                #Environment update: next state
                actual_month = new_month + int(timestep / (30*24*60))
                next_state, reward, game_over = env.update_env(direction, energy_ai, max_energy, actual_month, timestep)
                total_reward += reward
                          
                #Storing Transition in Memory
                dqn.remember([current_state, action, reward, next_state, r_hat], game_over)
                
                #Gathering Inputs and Targets in separate Batches
                inputs, targets = dqn.get_batch(model, batch_size)             
                
                #Compute the loss over the all Batches 
                loss += model.train_on_batch(inputs, targets)
                
                # s --> s'
                current_state = next_state
                
                #Performance metrics
                # inrange time
                if env.temperature_ai >= optimal_temperature[0] and env.temperature_ai <= optimal_temperature[1] :
                    t_in_ai += 1
                if env.temperature_noai >= optimal_temperature[0]  and env.temperature_noai <= optimal_temperature[1] :
                    t_in_noai += 1   
                
                # mse from optimal T = 21°
                mse_T_ai += ((env.temperature_ai - 21)**2)**(1/2)
                mse_T_noai += ((env.temperature_noai - 21)**2)**(1/2)
            else:
                break
    
        #Printing training result for each Epoch
        print("\n\n")
        print("Epoch: {:03d}/{:03d} (t = {}')".format(epoch, number_epochs, timestep))
        print("Energy spent with an AI: {:.0f}".format(env.total_energy_ai))
        print("Energy spent with No AI: {:.0f}".format(env.total_energy_noai))
        print("\nTime in range AI: {:.2f}".format(t_in_ai/timestep))
        print("Time in range No AI: {:.2f}".format(t_in_noai/timestep))
        print("\nTemperature mse AI: {:.2f}".format(mse_T_ai/timestep))
        print("Temperature mse No AI: {:.2f}".format(mse_T_noai/timestep))
        print("\nR_tot: {:.2f}, R_mean: {:.2f}, R_hat: {:.2f}".format(total_reward, total_reward/timestep, r_hat))
        print("Cost_tot: {:.2f}, Cost_mean: {:.2f}".format(loss, loss/timestep*100))
        print("Learning Rate: {:.2E}".format(K.eval((brain.model.optimizer.lr))))
        
        #Performance plot
        rew_plot.append(total_reward)
        AVG_rew_plot.append(total_reward/timestep)
        AVG_rew_plot_2.append((total_reward+10)/timestep)
        losses_plot.append(loss)
        AVG_losses_plot.append(loss/timestep)
        if epoch % 25 == 0:
            plt.subplot(2,3,1)
            plt.plot(rew_plot)
            plt.xlabel("epochs")
            plt.ylabel("r")
            plt.title("Reward")
            
            plt.subplot(2,3,2)
            plt.plot(AVG_rew_plot)
            plt.xlabel("epochs")
            plt.ylabel("r_avg")      
            plt.title("Relative reward")
            
            plt.subplot(2,3,3)
            plt.plot(AVG_rew_plot_2)
            plt.xlabel("epochs")
            plt.ylabel("r_avg")
            plt.title("Relative Reward - (no Rend)")
            
            plt.subplot(2,3,4)
            plt.plot(losses_plot)
            plt.xlabel("epochs")
            plt.ylabel("J")
            plt.title("Cost")
            
            plt.subplot(2,3,5)
            plt.plot(AVG_losses_plot)
            plt.xlabel("epochs")
            plt.ylabel("J_avg")
            plt.title("Relative Cost")
            
            plt.subplot(2,3,6)
            plt.plot(r_hat_plot)
            plt.xlabel("epochs")
            plt.ylabel("R_hat")
            plt.title("Reward hat")
            
        #Saving the model
        model.save("modelBVSO.h5")
        batch_memory = dqn.memory 
        last_lr = K.eval((brain.model.optimizer.lr))
        with open("memory.pickle","wb") as f:
            pickle.dump([batch_memory, epoch, rew_plot, AVG_rew_plot, AVG_rew_plot_2, losses_plot, AVG_losses_plot, last_lr], f)





