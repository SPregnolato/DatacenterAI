#TRAINING THE AI

#Installing Keras
# conda install -c conda-forge keras

#Libraries
import numpy as np
import DQL_environment
import DQL_agent_PyTorch
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import keras.backend as K
import pickle
import torch


# Setting Parameters
number_epochs = 1000
epoch_len = 2 * 30 * 24 * 60  
 

max_memory = 2**12
batch_size = 2**10

r_hat = -0.2
beta = 0.01 # avg reward step --> consider 0.001
discount = 1 # discount factor
tau_soft = .2 #temperature softmax

number_actions = 7
direction_boundary = (number_actions - 1) / 2
temperature_step = 1.5
max_energy = direction_boundary * temperature_step
optimal_temperature = (20.0, 24.0)
avg_optimal_temperature = np.average(optimal_temperature)

# Building the environment
env = DQL_environment.Environment(optimal_temperature = optimal_temperature, initial_month = 0, initial_number_users = 20, initial_rate_data = 30, max_energy = max_energy)
env.train = True
current_state, _, _ = env.observe()
number_states = current_state.shape[1]

# # Building the Policy (neural network)
# model = DQL_agent.Brain(learning_rate, number_actions, number_states,  loss_f, opt).model
 
#Building the RL algorithm
dqn = DQL_agent_PyTorch.DQN(max_memory = max_memory, discount = discount, n_actions = number_actions, n_states = number_states)
model = dqn.model
#Training the AI
timestep_max = 0
rew_plot = []
AVG_rew_plot = []
AVG_rew_plot_2 = []
losses_plot = []
epoch_plot = []
AVG_losses_plot = []
r_hat_plot = []
performance_plot = []
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
        mse_T_ai = []
        mse_T_noai = []
    
        # Loop over Timesteps (1 Timestep = 1 Minute) in one Epoch
        for timestep in range(epoch_len):
            if not game_over:
                
                # Choose action a (softmax) + AVG Q
                action, q_hat, direction, energy_ai = DQL_agent_PyTorch.policy(model, current_state, tau_soft, number_actions, direction_boundary, temperature_step)
                    
                # Environment update: next state
                actual_month = new_month + int(timestep / (30*24*60))
                next_state, reward, game_over = env.update_env(direction, energy_ai, max_energy, actual_month, timestep)
                total_reward += reward
                 
                # AVG reward update
                # Numpy to Torch
                next_state_t = torch.from_numpy(next_state).float()
                # prediction
                next_q_hats_t = model(next_state_t) 
                #Torch to numpy
                next_q_hats = next_q_hats_t.detach().numpy().squeeze()
                next_q_hat = np.max(next_q_hats)
                delta = reward - r_hat + discount * next_q_hat - q_hat
                r_hat += beta * delta
                
                
                dqn.update(reward, current_state, next_state, action, batch_size)
                


                # s --> s'
                current_state = next_state
                
                # Performance metrics
                # inrange time
                if env.temperature_ai >= optimal_temperature[0] and env.temperature_ai <= optimal_temperature[1] :
                    t_in_ai += 1
                if env.temperature_noai >= optimal_temperature[0]  and env.temperature_noai <= optimal_temperature[1] :
                    t_in_noai += 1   
                
                # mse from optimal T = 22°
                mse_T_ai.append(env.temperature_ai - avg_optimal_temperature)              
                mse_T_noai.append(env.temperature_noai - avg_optimal_temperature)
            else:
                break
    
        #Printing training result for each Epoch
        pidx = np.tanh((env.total_energy_noai+1)/(env.total_energy_ai+1)) + np.tanh((t_in_ai+1)/(t_in_noai+1)) -2
        mse_T_ai = np.linalg.norm(mse_T_ai)
        mse_T_noai = np.linalg.norm(mse_T_noai)
        print("\n\n")
        print("Epoch: {:03d}/{:03d} (t = {}', R_tot: {:.2f})".format(epoch, number_epochs, timestep, total_reward))
        print("Energy spent with an AI: {:.0f}".format(env.total_energy_ai))
        print("Energy spent with No AI: {:.0f}".format(env.total_energy_noai))
        print("\nTime in range AI: {:.2f}".format(t_in_ai/timestep))
        print("Time in range No AI: {:.2f}".format(t_in_noai/timestep))
        print("\nTemperature mse AI: {:.2f}".format(mse_T_ai/timestep))
        print("Temperature mse No AI: {:.2f}".format(mse_T_noai/timestep))
        print("\nR_mean: {:.2f}, R_hat: {:.2f}".format(total_reward/timestep*10, r_hat*10))
        print("J_mean: {:.3f}".format(loss/timestep*100))
        print("Performance: {:.2f}".format(pidx))
        # Max Model
        if timestep > timestep_max:
            model.save("modelBVSOmax.h5")
            timestep_max = timestep
            
            
        #Performance plot
        rew_plot.append(total_reward)
        AVG_rew_plot.append(total_reward/timestep)
        AVG_rew_plot_2.append((total_reward-reward)/timestep)
        epoch_plot.append(timestep)
        AVG_losses_plot.append(loss/timestep)
        losses_plot.append(loss)
        r_hat_plot.append(r_hat)
        performance_plot.append(pidx)
        if epoch % 25 == 0:
            model.save("modelBVSO"+str(epoch)+".h5")
            
            plt.subplot(3,3,1)
            plt.plot(rew_plot)
            plt.xlabel("epoch")
            plt.ylabel("r")
            plt.title("Reward")
            
            plt.subplot(3,3,2)
            plt.plot(AVG_rew_plot)
            plt.xlabel("epoch")
            plt.ylabel("r_avg")      
            plt.title("Relative reward")
            
            plt.subplot(3,3,3)
            plt.plot(AVG_rew_plot_2)
            plt.xlabel("epoch")
            plt.ylabel("r_avg2")
            plt.title("Relative Reward - (no Rend)")
            
            plt.subplot(3,3,4)
            plt.plot(epoch_plot)
            plt.xlabel("epoch")
            plt.ylabel("timesteps")
            plt.title("Episode Length")
            
            plt.subplot(3,3,5)
            plt.plot(AVG_losses_plot)
            plt.xlabel("epochs")
            plt.ylabel("J_avg")
            plt.title("Relative Cost")
            
            plt.subplot(3,3,6)
            plt.plot(r_hat_plot)
            plt.xlabel("epoch")
            plt.ylabel("R_hat")
            plt.title("Reward hat")
            
            plt.subplot(3,3,7)
            plt.plot(performance_plot)
            plt.xlabel("epoch")
            plt.ylabel("Pidx")
            plt.title("Performance Index")
            
            
        #Saving the model
        model.save("modelBVSO.h5")
        batch_memory = dqn.memory 
        last_lr = K.eval((model.optimizer.lr))
        with open("memory.pickle","wb") as f:
            pickle.dump([batch_memory, epoch, rew_plot, AVG_rew_plot, AVG_rew_plot_2, epoch_plot, AVG_losses_plot, r_hat_plot, performance_plot, losses_plot, last_lr], f)





