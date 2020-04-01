
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


#Setting Parameters
beta = 0.01 # reward step
number_epochs = 1000
max_memory = 3000
batch_size = 300

number_actions = 7
direction_boundary = (number_actions - 1) / 2
temperature_step = 1.5
max_energy = direction_boundary * temperature_step
optimal_temperature = (20.0, 24.0)

#Building the environment
env = DQL_environment.Environment(optimal_temperature = optimal_temperature, initial_month = 0, initial_number_users = 20, initial_rate_data = 30, max_energy = max_energy)

#Building the brain
learning_rate = 0.01
loss_f = 'mean_squared_logarithmic_error'
brain = DQL_brain.Brain(learning_rate = learning_rate, number_actions = number_actions, loss = loss_f)

#Building the model
dqn = DQL_dqn.DQN(max_memory = max_memory, discount = 0.99)


#Training Mode = on
train = True

#Training the AI
env.train = train
model = brain.model
early_stopping = False
patience = 10
best_total_reward = -np.inf
patience_count = 0
rew_plot = []
AVG_rew_plot = []
AVG_rew_plot_2 = []
losses_plot = []
AVG_losses_plot = []
plt.figure()
if (env.train):
    # Loop over Epochs (1 Epoch = 2 Months)
    for epoch in range(1, number_epochs):
        total_reward = 0
        loss = 0.
        new_month = np.random.randint(0, 12)
        env.reset(new_month = new_month)
        game_over = False
        current_state, _, _ = env.observe()
        timestep = 0
        r_hat = 0 
        t_in_ai = 0
        t_in_noai = 0
        mse_T_ai = 0
        mse_T_noai = 0
# review the leraning rate decay <---------------------------------------
        # #learning rate update
        # if epoch % 100 == 0:
        #     brain.learning_rate /= 10 
        #     model.compile(loss = loss_f, optimizer = Adam(lr = brain.learning_rate))
# ------------------------------------------------------------------------    
        #Loop over Timesteps (1 Timestep = 1 Minute) in one Epoch
        for timestep in range(2 * 30 * 24 * 60):
            if not game_over:
                #Choose action a (softmax)
                q_values = model.predict(current_state)
                action = np.random.choice(number_actions, p = q_values[0])      
                if (action - direction_boundary < 0):
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temperature_step
                    
                #Environment update: next state
                next_state, reward, game_over = env.update_env(direction, energy_ai, max_energy, int(timestep / (30*24*60)))
                total_reward += reward
                
                #AVG reward
                q_hat = q_values[0][action]
                next_q_hat = max(model.predict(next_state)[0])
                delta = reward - r_hat + next_q_hat - q_hat
                r_hat += beta * delta
                
                #Storing Transition in Memory
                dqn.remember([current_state, action, reward, next_state], game_over)
                
                #Gathering Inputs and Targets in separate Batches
                inputs, targets = dqn.get_batch(model, batch_size = batch_size)
                
                #Compute the loss over the all Batches 
                loss += model.train_on_batch(inputs, targets)
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
        print("Loss_tot: {:.2f}, Loss_mean: {:.2f}".format(loss, loss/timestep*100))
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
        model.save("modelBVSO.h5")
        
        
        rew_plot.append(total_reward)
        AVG_rew_plot.append(total_reward/timestep)
        AVG_rew_plot_2.append((total_reward+10)/timestep)
        losses_plot.append(loss)
        AVG_losses_plot.append(loss/timestep)
        if epoch % 25 == 0:
            plt.subplot(2,3,1)
            plt.plot(rew_plot)
            plt.xlabel("epochs")
            plt.ylabel("reward")
            plt.title("Episode Reward")
            plt.subplot(2,3,2)
            plt.plot(AVG_rew_plot)
            plt.xlabel("epochs")
            plt.ylabel("AVG reward")
            plt.title("Relative Episode Reward")
            plt.subplot(2,3,3)
            plt.plot(AVG_rew_plot_2)
            plt.xlabel("epochs")
            plt.ylabel("AVG reward")
            plt.title("Relative Episode Reward - (no Rend)")
            plt.subplot(2,3,4)
            plt.plot(losses_plot)
            plt.xlabel("epochs")
            plt.ylabel("loss")
            plt.title("Episode Training Loss")
            plt.subplot(2,3,5)
            plt.plot(AVG_losses_plot)
            plt.xlabel("epochs")
            plt.ylabel("Episode Relative Training Loss")
            plt.title("Model Training")





