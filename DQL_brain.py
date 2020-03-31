
#BUILDING THE BRAIN
#Import Keras libray: from modules import classes
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam

#Brain Class --> to create as many objects as we want
class Brain(object):
    
    def __init__ (self, learning_rate = 0.001, number_actions = 5, number_states = 3):
        
        self.learning_rate = learning_rate 
        
        #Input layer
        states = Input(shape = (number_states,))
        
        #Hidden layers
        x = Dense(units = 64, activation = 'relu')(states)
        x = Dropout(rate = 0.1)(x)
        y = Dense(units = 32, activation = 'relu')(x)
        y = Dropout(rate = 0.1)(y)
        #Output layer
        q_values = Dense(units = number_actions, activation = 'softmax')(y)
        
        #Assembling the full architecture in a model object (object variable)
        self.model = Model(inputs = states, outputs = q_values)
        
        #Compiling the model with loss and optimizer (applying the compile method)
        self.model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate))