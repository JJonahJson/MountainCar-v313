import numpy as np
import tensorflow as tf
from net_model import NetModel
from training_mode import TrainingMode

class DQNAgent:
    
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.min_experiences = min_experiences
        self.max_experiences = max_experiences
        self.optimizer = tf.optimizers.Adam(lr)
        self.model = NetModel(num_states, hidden_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        
    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))

    def train(self, target_net, chosen_training):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        
        
        if chosen_training is TrainingMode.DoubleDQN:
            values = np.array(target_net.predict(states_next))[range(self.batch_size), np.argmax(self.predict(states_next), axis=1)]
            actual_values = np.where(dones, rewards, rewards + self.gamma * values)

        else:
            value_next = np.max(target_net.predict(states_next), axis=1)
            actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        
        return loss

    def get_action_epsilon_greedy(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, train_net):
        variables1 = self.model.trainable_variables
        variables2 = train_net.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

    def soft_update_weights(self, train_net):
        tau = 0.1
        q_network_theta = train_net.model.get_weights()
        target_network_theta = self.model.get_weights()
        counter = 0
        for q_weight, target_weight in zip(q_network_theta, target_network_theta):
            target_weight = target_weight * (1 - tau) + q_weight * tau
            target_network_theta[counter] = target_weight
            counter += 1
        self.model.set_weights(target_network_theta)
