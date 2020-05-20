import pickle
import matplotlib.pyplot as plt
from keras.models import load_model

# open a file, where you stored the pickled data
file = open('memory.pickle', 'rb')

# dump information to that file
# pickle.dump([batch_memory, epoch, rew_plot, AVG_rew_plot, AVG_rew_plot_2, epoch_plot, AVG_losses_plot, r_hat_plot, performance_plot, losses_plot, davg_plot, last_lr], f)
data = pickle.load(file)

batch_memory = data[0]
rew_plot =data[2] 
AVG_rew_plot =data[3] 
AVG_rew_plot_2 = data[4] 
epoch_plot = data[5]
AVG_losses_plot = data[6]
r_hat_plot = data[7]
performance_plot = data[8]
losses_plot = data[9]
davg_plot = data[10]

#Loading pre trained model (parameters: weights)
# model_name = "modelBVSO800.h5"
# model = load_model(model_name)


# print(model.predict(batch_memory[10][0][0])[0])
# print(model.predict(batch_memory[100][0][0])[0])

plt.figure()
plt.subplot(3,3,1)
plt.plot(rew_plot)
plt.xlabel("epochs")
plt.ylabel("r")
plt.title("Reward")

plt.subplot(3,3,2)
plt.plot(AVG_rew_plot)
plt.xlabel("epochs")
plt.ylabel("r_avg")      
plt.title("Relative reward")

plt.subplot(3,3,3)
plt.plot(AVG_rew_plot_2)
plt.xlabel("epochs")
plt.ylabel("r_avg2")
plt.title("Relative Reward - (no Rend)")

plt.subplot(3,3,4)
plt.plot(epoch_plot)
plt.xlabel("epochs")
plt.ylabel("timesteps")
plt.title("Episode Length")

plt.subplot(3,3,5)
plt.plot(AVG_losses_plot)
plt.xlabel("epochs")
plt.ylabel("J_avg")
plt.title("Relative Cost")

plt.subplot(3,3,6)
plt.plot(r_hat_plot)
plt.xlabel("epochs")
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
            



# close the file
file.close()