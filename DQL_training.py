#TRAINING THE AI

#Installing Keras
# conda install -c conda-forge keras

#Libraries
import numpy as np
import DQL_environment
import DQL_agent
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Model
from keras.models import load_model
import keras.backend as K
import pickle

# Setting Parameters
number_epochs = 10000
epoch0 = 1
epoch_len = 2 * 30 * 24 * 60  
 
learning_rate = 0.01
decay = 1e-8
loss_f = 'huber_loss'  # huber_loss <---- check delta parameter
opt = Adam(learning_rate=learning_rate, decay = decay, beta_1=0.9, beta_2=0.999, amsgrad=False)

max_memory = 2**13
batch_size = 2**10  # full batch
mini_batch_size = 2**7


r_hat = -0.1
beta = 0.01 # avg reward step --> consider 0.001
discount = 1 # discount factor
tau_soft0 = 1 #temperature softmax
eps0 = 1 # epsilon greedy

number_actions = 7
direction_boundary = (number_actions - 1) / 2
temperature_step = 1.5
max_energy = direction_boundary * temperature_step
optimal_temperature = (20.0, 24.0)


# Building the environment
env = DQL_environment.Environment(optimal_temperature = optimal_temperature, initial_month = 0, initial_number_users = 20, initial_rate_data = 30, max_energy = max_energy)
env.train = True
current_state, _, _ = env.observe()
number_states = current_state.shape[1]


# # Building the Policy (neural network)
model = DQL_agent.Brain(learning_rate, number_actions, number_states,  loss_f, opt).model
config = model.get_config()
weights = model.get_weights()
target_model = Model.from_config(config)
target_model.set_weights(weights)
c_steps = 2 # steps before copying model in target

#Building the RL algorithm
dqn = DQL_agent.DQN(max_memory = max_memory, discount = discount)

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
davg_plot =[]

# CONTINUE TRAINING
continue_training = False
if continue_training:
    # open a file, where you stored the pickled data
    file = open('memory.pickle', 'rb')
    data = pickle.load(file)
    # pickle.dump([batch_memory, epoch, rew_plot, AVG_rew_plot, AVG_rew_plot_2, epoch_plot, AVG_losses_plot, r_hat_plot, performance_plot, losses_plot, davg_plot, last_lr], f)
    dqn.memory = data[0]
    epoch0 = 676
    #Loading pre trained model (parameters: weights)
    model_name = "modelBVSO675.h5"
    model = load_model(model_name)
    rew_plot =data[2] 
    AVG_rew_plot =data[3] 
    AVG_rew_plot_2 = data[4] 
    epoch_plot = data[5]
    AVG_losses_plot = data[6]
    r_hat_plot = data[7]
    losses_plot = data[8]
    davg_plot = data[9]


plt.figure()
t_count = 0
if (env.train):
    # Loop over Epochs (1 Epoch = 2 Months)
    for epoch in range(epoch0, number_epochs+1):
        total_reward = 0
        loss = 0.
        new_month = np.random.randint(0, 12-int(epoch_len / (30*24*60)))
        env.reset(new_month = new_month)
        game_over = False
        current_state, _, _ = env.observe()
        t_in_ai = 0
        t_in_noai = 0
        mse_T_ai = []
        mse_T_noai = []
    
        # Loop over Timesteps (1 Timestep = 1 Minute) in one Epoch
        for timestep in range(epoch_len):
            if not game_over:
                t_count += 1
                
                # Choose action a (softmax) + AVG Q
                # tau_soft = (tau_soft0 - t_count / 1000)**3
                # if tau_soft < 0.1 :
                #     tau_soft = 0.1 
                #
                # action, q_hat, direction, energy_ai, q_values, q_values_norm, probs = DQL_agent.policy_softmax(model, current_state, tau_soft, number_actions, direction_boundary, temperature_step)                
                
                eps = (eps0 - t_count / 5000)**3
                if eps < 0.05 * number_actions:
                    eps = 0.05 * number_actions
                action, q_hat, direction, energy_ai, q_values, probs = DQL_agent.policy_greedy(model, current_state, eps , number_actions, direction_boundary, temperature_step)
                
                
                # Environment update: next state
                actual_month = new_month + int(timestep / (30*24*60))
                next_state, reward, game_over = env.update_env(direction, energy_ai, actual_month, timestep)
                total_reward += reward
                
                
                
                # AVG reward update
                # next_q_hat = np.max(model.predict(next_state)[0])
                next_q_hat = np.max(target_model.predict(next_state)[0])
                delta = reward - r_hat + discount * next_q_hat - q_hat
                r_hat += beta * delta
      
                
                if timestep > 0:
                    # Storing Transition in Memory
                    dqn.remember([current_state, action, reward, next_state, r_hat], game_over)
                
                
                    # Gathering Inputs and Targets in separate Batches
                    # inputs, targets = dqn.get_batch(model, batch_size, r_hat)    
                    inputs, targets = dqn.get_batch(model, target_model, batch_size, r_hat) 
                    
               
                    # Compute the loss over the all Batches 
                    n_batches = len(inputs) // mini_batch_size
                    if n_batches < 2:
                        loss_batch = model.train_on_batch(inputs, targets)
                    else:
                        for i in range(n_batches):
                            loss_batch = model.train_on_batch(inputs[i*mini_batch_size:(i+1)*mini_batch_size], targets[i*mini_batch_size:(i+1)*mini_batch_size])                                                             
    
                    loss += loss_batch
                    
                    
                
                # s <-- s'
                prev_state = current_state
                current_state = next_state
                
                
                # model --> target copy (every c_steps)
                if (t_count) % c_steps == 0:
                    config = model.get_config()
                    weights = model.get_weights()
                    target_model = Model.from_config(config)
                    target_model.set_weights(weights)
                    # print("-----model update----")
                    
                
                # Performance metrics
                # inrange time
                if env.temperature_ai >= optimal_temperature[0] and env.temperature_ai <= optimal_temperature[1] :
                    t_in_ai += 1
                if env.temperature_noai >= optimal_temperature[0]  and env.temperature_noai <= optimal_temperature[1] :
                    t_in_noai += 1   
                
                # mse from optimal T = 22°
                mse_T_ai.append(env.temperature_ai - env.avg_optimal_temperature)              
                mse_T_noai.append(env.temperature_noai - env.avg_optimal_temperature)
            else:
                break
    
        #Printing training result for each Epoch
        pidx = ( np.tanh((env.total_energy_noai+1)/(env.total_energy_ai+1)) + np.tanh((t_in_ai+1)/(t_in_noai+1)) + 2 ) * ( timestep / epoch_len )**(0.25)
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
        print("J_mean: {:.3f}, batches: {}/{}".format(loss/timestep*100, n_batches, int(batch_size/mini_batch_size)))
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
        davg_plot.append(total_reward/timestep - r_hat)
        
        if epoch % 25 == 0:
            
            print(f'current_state: {prev_state}')
            print(f'q_values: {q_values}')
            print(f'probs= {probs}')
            
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
            
            plt.subplot(3,3,8)
            plt.plot(davg_plot)
            plt.xlabel("epoch")
            plt.ylabel("DAVG")
            plt.title("r_avg-r_hat")
            
            
            
            
        #Saving the model
        model.save("modelBVSO.h5")
        batch_memory = dqn.memory 
        last_lr = K.eval((model.optimizer.lr))
        with open("memory.pickle","wb") as f:
            pickle.dump([batch_memory, epoch, rew_plot, AVG_rew_plot, AVG_rew_plot_2, epoch_plot, AVG_losses_plot, r_hat_plot, performance_plot, losses_plot, davg_plot, last_lr], f)





