# base environment
#BUILDING THE ENVIRONMENT
#Libraries
import numpy as np

#environment class
class Environment(object):
    
    #Introducing and Initializing all the parametersand variables of the environment
    def __init__ (self,
                  optimal_temperature = (20.0, 24.0),
                  initial_month = 0,
                  initial_number_users = 10,
                  initial_rate_data = 60,
                  max_energy = 5):
        self.monthly_atmospheric_temperatures = [1.0, 5.0, 7.0, 10.0, 11.0, 20.0, 23.0, 24.0, 22.0, 10.0, 5.0, 1.0]
        self.initial_month = initial_month
        self.atmospheric_temperature = \
        self.monthly_atmospheric_temperatures[initial_month]
        self.optimal_temperature = optimal_temperature
        self.min_temperature = optimal_temperature[0] - (max_energy + 2)
        self.max_temperature = optimal_temperature[1] + (max_energy + 2)
        self.min_number_users = 10
        self.max_number_users = 100
        self.max_update_users = 5
        self.min_rate_data = 20
        self.max_rate_data = 300
        self.max_update_data = 10
        self.initial_number_users = initial_number_users
        self.current_number_users = initial_number_users
        self.initial_rate_data = initial_rate_data
        self.current_rate_data = initial_rate_data
        self.kmodel = 0.3
        self.intrinsic_temperature = self.atmospheric_temperature + self.kmodel  * (1.25 * self.current_number_users + 1.25 * self.current_rate_data)
        self.temperature_ai = (self.optimal_temperature[0] + self.optimal_temperature[1]) / 2.0
        self.temperature_noai = (self.optimal_temperature[0] + self.optimal_temperature[1]) / 2.0
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1
        self.range_error = 0
        self.delta_intrinsic_temperature = 0

    # Method to update the environment object right after the AI (agent) plays the action
    def update_env(self, direction, energy_ai, max_energy, month, timestep):
        
        # energy_no_ai 
        energy_noai = 0
        if (self.temperature_noai < self.optimal_temperature[0]):
            energy_noai = self.optimal_temperature[0] - self.temperature_noai
            if energy_noai > max_energy:
                energy_noai = max_energy
            self.temperature_noai = self.temperature_noai + energy_noai
        elif (self.temperature_noai > self.optimal_temperature[1]):
            energy_noai = self.temperature_noai - self.optimal_temperature[1]
            if energy_noai > max_energy:
                energy_noai = max_energy
            self.temperature_noai = self.temperature_noai - energy_noai

        
        # Next state: S'
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[month]
        self.current_number_users += np.random.randint(-self.max_update_users, self.max_update_users)
        if (self.current_number_users > self.max_number_users):
            self.current_number_users = self.max_number_users
        elif (self.current_number_users < self.min_number_users):
            self.current_number_users = self.min_number_users
        self.current_rate_data += np.random.randint(-self.max_update_data, self.max_update_data)
        if (self.current_rate_data > self.max_rate_data):
            self.current_rate_data = self.max_rate_data
        elif (self.current_rate_data < self.min_rate_data):
            self.current_rate_data = self.min_rate_data
        past_intrinsic_temperature = self.intrinsic_temperature
    
        self.intrinsic_temperature = self.atmospheric_temperature + self.kmodel  * (1.25 * self.current_number_users + 1.25 * self.current_rate_data)
        self.delta_intrinsic_temperature = self.intrinsic_temperature - past_intrinsic_temperature
        if (direction == -1):
            delta_temperature_ai = -energy_ai
        elif (direction == 1):
            delta_temperature_ai = energy_ai
        self.temperature_ai += self.delta_intrinsic_temperature + delta_temperature_ai
        self.temperature_noai += self.delta_intrinsic_temperature
        
        # Reward
        if (self.temperature_ai >= self.optimal_temperature[0]) and (self.temperature_ai <= self.optimal_temperature[1]):
            
            # reward ideas
            self.reward = 2 - (energy_ai / max_energy)
            # self.reward = 2 - (energy_ai / (max_energy))
            # self.reward = 2 - (energy_ai / (2*max_energy)) + np.clip(np.log((self.total_energy_noai+1) / (self.total_energy_ai+1)), -0.5, 0.5)
            # self.reward = 2 - (energy_ai / (2*max_energy)) - (self.total_energy_ai / ((timestep+1) * max_energy))
        else:
            #reward ideas
            self.reward = -1 - (energy_ai / max_energy) 
            # self.reward = -0.1 
            # self.reward = -0.1 - (energy_ai / (4*max_energy))
           
        
        # Exit Conditions 
        if (self.temperature_ai < self.min_temperature):
            if (self.train == 1):
                self.game_over = 1
                # self.reward = 0
                self.reward = -2
            else:
                # print('Error - Tmin')
                self.range_error += 1
                self.temperature_ai = self.optimal_temperature[0]
                self.total_energy_ai += self.optimal_temperature[0] - self.temperature_ai
        elif (self.temperature_ai > self.max_temperature):
            if (self.train == 1):
                self.game_over = 1
                # self.reward = 0
                self.reward = -2
            else:
                # print('Error - Tmax')
                self.range_error += 1
                self.temperature_ai = self.optimal_temperature[1] 
                self.total_energy_ai += self.temperature_ai - self.optimal_temperature[1]
     
        # Scores
        self.total_energy_ai += energy_ai
        self.total_energy_noai += energy_noai
    
        # Scaling the next state (normalize the inputs of the neural network)
        scaled_temperature_ai = (self.temperature_ai - self.min_temperature) / (self.max_temperature - self.min_temperature)
        scaled_number_users = (self.current_number_users - self.min_number_users) / (self.max_number_users - self.min_number_users)
        scaled_rate_data = (self.current_rate_data - self.min_rate_data) / (self.max_rate_data - self.min_rate_data)
        next_state = np.matrix( [scaled_temperature_ai, scaled_number_users, scaled_rate_data] )
        
        #Returning: reward, next state, exit condition
        return next_state, self.reward, self.game_over
    
    
    # Method to reset the environment
    def reset(self, new_month):
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[new_month]
        self.initial_month = new_month
        self.current_number_users = self.initial_number_users
        self.current_rate_data = self.initial_rate_data
        self.intrinsic_temperature = self.atmospheric_temperature + self.kmodel  * (1.25 * self.current_number_users + 1.25 * self.current_rate_data)
        self.temperature_ai =  (self.optimal_temperature[0] + self.optimal_temperature[1]) / 2.0
        self.temperature_noai = (self.optimal_temperature[0] + self.optimal_temperature[1]) / 2.0
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1
        self.range_error = 0
        self.delta_intrinsic_temperature = 0
       
      
    # Method to extract: current state, last reward, exit condition
    def observe(self):
        scaled_temperature_ai = (self.temperature_ai - self.min_temperature) / (self.max_temperature - self.min_temperature)
        scaled_number_users = (self.current_number_users - self.min_number_users) / (self.max_number_users - self.min_number_users)
        scaled_rate_data = (self.current_rate_data - self.min_rate_data) / (self.max_rate_data - self.min_rate_data)
        current_state = np.matrix([scaled_temperature_ai, scaled_number_users, scaled_rate_data])
        return current_state, self.reward, self.game_over




    
    
    
    
    
    
    
    
