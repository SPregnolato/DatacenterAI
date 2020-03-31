
# TESTING the AI

#Installing Keras
# conda install -c conda-forge keras
#Libraries

import os
import numpy as np
import random as rn
from keras.models import load_model
import DQL_environment
import matplotlib.pyplot as plt


#Setting seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

#Setting parameters
number_actions = 5
direction_boundary = (number_actions - 1) / 2
temperature_step = 1.5

#Building the environment
env = DQL_environment.Environment(optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)

#Loading pre trained model (parameters: weights)
model = load_model("model.h5")

#Inference mode
train = False

#1-Year simulation in Inference Mode
env.train = train
current_state, _, _ = env.observe()
actions = []
AI_T = []
NO_AI_T = []
time = []
inside_AI = 0
inside_NO_AI = 0
E_AI = []
E_NO_AI = []
for timestep in range(0, 12 * 30 * 24 * 60):
#for timestep in range(0, 12 * 30 ):
    q_values = model.predict(current_state)
    action = np.argmax(q_values[0])
    actions.append((action - direction_boundary)*temperature_step)
    if (action - direction_boundary < 0):
        direction = -1
    else:
        direction = 1
    energy_ai = abs(action - direction_boundary) * temperature_step
    next_state, reward, game_over, ai_T, no_ai_T = env.update_env(direction, energy_ai, int(timestep / (30*24*60)))
    AI_T.append(ai_T)
    NO_AI_T.append(no_ai_T)
    E_AI.append(env.total_energy_ai)
    E_NO_AI.append(env.total_energy_noai)
    time.append(timestep)
    current_state = next_state
    if ai_T > (17) and ai_T < (25):
        inside_AI += 1
    if no_ai_T > (17) and no_ai_T < (25):
        inside_NO_AI += 1
    print('Status: {}%'.format(100*timestep/(12 * 30 * 24 * 60)))





plt.figure()
plt.subplot(3,2,1)
plt.grid(True)
plt.plot(AI_T)
plt.plot(NO_AI_T)
plt.hlines(18, 0, len(time), colors='k', linestyles='dashed')
plt.hlines(24, 0, len(time), colors='k', linestyles='dashed')
plt.title('Server Temperature Comparison')
plt.legend(('Original AI', 'No AI'))
plt.xlabel('time [min]')
plt.ylabel('T_server [°C]')

plt.subplot(3,2,3)
plt.grid(True)
plt.plot(E_AI)
plt.plot(E_NO_AI)
plt.title('Energy Consumption')
plt.legend(('Original AI', 'No AI'))
plt.xlabel('time [min]')
plt.ylabel('Energy')



plt.subplot(3,2,2)
plt.grid(True)
plt.plot(actions)
plt.title('Actions')
plt.xlabel('time [min]')
plt.ylabel('DT [°C]')

plt.subplot(3,2,5)
plt.grid(True)
plt.bar([0] , [env.total_energy_ai])
plt.bar([1] , [env.total_energy_noai])
plt.xticks([0, 1], ('Original AI', 'No AI'))
plt.title('Total Energy Consumption\n (the less the better)')
plt.legend(('Original AI', 'No AI'))

plt.subplot(3,2,4)
plt.grid(True)
plt.bar([0] , [inside_AI/len(time)])
plt.bar([1] , [inside_NO_AI/len(time)])
plt.xticks([0, 1], ('Original AI', 'No AI'))
plt.title('InRange Time\n (the more the better)')
plt.legend(('Original AI', 'No AI'))
plt.ylabel('[%]')

    
#Printing the result
print("\n")
print("Total Energy spent with an AI: {:.0f}".format(env.total_energy_ai))
print("Total Energy spent with no AI: {:.0f}".format(env.total_energy_noai))
print("ENERGY SAVED: {:.0f} %".format((env.total_energy_noai - env.total_energy_ai) / env.total_energy_noai * 100))






plt.figure()

plt.subplot(2,1,1)
plt.grid(True)
plt.plot(actions)
plt.title('Actions')
plt.xlabel('time [min]')
plt.ylabel('DT [°C]')

plt.subplot(2,1,2)
plt.grid(True)
plt.bar([0] , [env.total_energy_ai])
plt.bar([1] , [env.total_energy_noai])
plt.xticks([0, 1], ('Original AI', 'No AI'))
plt.title('Total Energy Consumption\n (the less the better)')
plt.legend(('Original AI', 'No AI'))
