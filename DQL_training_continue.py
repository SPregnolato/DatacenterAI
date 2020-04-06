# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 00:07:09 2020

@author: spreg
"""


import pickle
import matplotlib.pyplot as plt

# open a file, where you stored the pickled data
file = open('memory.pickle', 'rb')

# dump information to that file
# pickle.dump([batch_memory, epoch, rew_plot, AVG_rew_plot, AVG_rew_plot_2, epoch_plot, AVG_losses_plot, r_hat_plot, losses_plot, last_lr], f)
data = pickle.load(file)
rew_plot =data[2] 
AVG_rew_plot =data[3] 
AVG_rew_plot_2 = data[4] 
epoch_plot = data[5]
AVG_losses_plot = data[6]
r_hat_plot = data[7]
losses_plot = data[8]

plt.figure()
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
plt.ylabel("r_avg2")
plt.title("Relative Reward - (no Rend)")

plt.subplot(2,3,4)
plt.plot(epoch_plot)
plt.xlabel("epochs")
plt.ylabel("timesteps")
plt.title("Episode Length")

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




# close the file
file.close()