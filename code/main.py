import os
import gym
import sys
import glob
import datetime
import argparse
import numpy as np
import tensorflow as tf
import printer
from dqn_agent import DQNAgent
from training_mode import TrainingMode

def play(env, epsilon, train_net, chosen_training, wanna_see, target_net = None, copy_step = None):
    step = 0
    rewards = 0
    done = False
    losses = list()
    goal_reached = False
    observations = env.reset()
    
    while not done:
        action = train_net.get_action_epsilon_greedy(observations, epsilon)
        prev_observations = observations
        observations, reward, done, _ = env.step(action)
        if wanna_see:
            env.render()
        
        if observations[0] >= 0.5:
            goal_reached = True

        rewards += reward
        if done:
            env.reset()

        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
        train_net.add_experience(exp)
        
        if chosen_training is not TrainingMode.DQN:
            loss = train_net.train(target_net, chosen_training)
        else:
            loss = train_net.train(train_net, chosen_training)
            
        if isinstance(loss, int):
            losses.append(loss)
        else:
            losses.append(loss.numpy())
        
        if chosen_training is not TrainingMode.DQN:
            step += 1
            if step % copy_step == 0:
                target_net.soft_update_weights(train_net)
                #target_net.copy_weights(train_net)

                
    return rewards, np.mean(losses), goal_reached

def main():
    #Add option to choose runtime to train or to play
    parser = argparse.ArgumentParser(description="MountainCar-v313 Train&Play")
    parser.add_argument("-m", "--mode", type=str, help="Type train or play", required=True,choices={"train", "play"})
    parser.add_argument("-p", "--personalize", help="Use this if you want to use your own hyperparameters", action='store_true')
    parser.add_argument("-s", "--see", help="Use this if you want to see the execution during the training", action='store_true')
    args = parser.parse_args()
    
    #Chosen environment
    env = gym.make('MountainCar-v0')
    num_states = len(env.observation_space.sample())
    num_actions = env.action_space.n

    #TRAINING MODE
    if args.mode == "train":
        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')

        # Variables for the training
        hidden_units = [24, 48]  # values represent the input of the hidden layer, the len() output is the number of hidden layers
        lr = 0.001
        gamma = 0.999
        batch_size = 256

        min_experiences = 100
        max_experiences = 400000

        epsilon = 1
        decay = 0.85
        min_epsilon = 0.01

        copy_step = 25
        training_cycles = 5000
        wanna_see = False
        
        try:
            print("\nINSERT THE NUMBER OF THE TRAINING MODE YOU WANT TO CHOOSE:\n")
            
            for mode in TrainingMode:
                print('%d: %s' %(mode.value, mode.name))
            print()
            chosen_training = TrainingMode(int(input()))
            print("\nYou've chose %s!\n" %(chosen_training.name))
            if chosen_training is TrainingMode.DQN:
                gamma = 0.98

            if args.personalize:
                print("\nINSERT THE LEARNING RATE (default 0.001): ")
                lr = float(input())
                print("\nINSERT THE BATCH SIZE (default 256): ")
                batch_size = int(input())
                print("\nINSERT THE MINIMUM EPSILON VALUE (default 0.01): ")
                min_epsilon = float(input())
                print("\nINSERT THE DECAY (default 0.85): ")
                decay = float(input())
                print("\nINSERT THE COPY STEP (default 25): ")
                copy_step = int(input())
                print("\nINSERT THE NUMBER OF TRAINING CYCLES (default 5000): ")
                training_cycles = int(input())
                print("\nINSERT THE MAXIMUM NUMBER OF SAVED EXPERIENCES (default 400000): ")
                max_experiences = int(input())
                print("\nINSERT THE GAMMA VALUE (default 0.999 per fixed Q targets e double DQN, consigliato 0.98 per DQN): ")
                gamma = float(input())
                print("\nINSERT THE NUMBER OF HIDDEN LAYERS (default 2): ")
                print("Remember that if later you'll want to play you can use only models with the default value of hidden layers, which is 2, with 24 neurons for the first and 48 for the second.\n If you want you can change the model in the play section")
                n_hidden = int(input())
                hidden_units = list()
                for i in range(1, n_hidden + 1):
                    print("\nINSERT THE NUMBER OF NEURONS IN HIDDEN LAYER %d: " %(i))
                    hidden_units.append(int(input()))
            if args.see:
                wanna_see = True
            
        except:
            print("Some problem occured, pay attention to the value you type!")
            sys.exit(1)
        
        #Variables for the tensorboard statistics
        current_time = datetime.datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
        log_dir = 'train_logs/' + chosen_training.name + '-' + current_time
        summary_writer = tf.summary.create_file_writer(log_dir)

        train_net = DQNAgent(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
        #The target net it's not required if we are using a simple DQN
        if chosen_training is not TrainingMode.DQN:
            target_net = DQNAgent(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)

        #Array with the rewards of each episode
        total_rewards = np.empty(training_cycles)
        total_rewards[0] = -200
        n_goal = 0

        print('\nLET\'S TRAIN!\n')
        try:
            for episode in range(1, training_cycles + 1):
                '''if episode == 1000:
                    train_net.batch_size = 128
                    if chosen_training is not TrainingMode.DQN:
                        target_net.batch_size = 128'''

                epsilon = max(min_epsilon, epsilon * decay)

                if chosen_training is not TrainingMode.DQN:
                    total_reward, loss_mean, goal_reached = play(env, epsilon, train_net, chosen_training, wanna_see, target_net, copy_step)
                else:
                    total_reward, loss_mean, goal_reached = play(env, epsilon, train_net, chosen_training, wanna_see)

                if wanna_see:
                    env.render()

                loss_mean = np.round(loss_mean, 5)
                if goal_reached:
                    n_goal += 1
                total_rewards[episode] = total_reward

                avg_rewards = np.round(total_rewards[max(0, episode - 100): episode].mean(), 5)
                '''if avg_rewards >= -110:
                    print('\nPROBLEM SOLVED!\n')
                    if chosen_training is not TrainingMode.DQN:
                        target_net.model.save_weights('./checkpoints/{}_target_{}goal_(-{})avg-reward.h5'.format(chosen_training.name, n_goal, np.round(avg_rewards)))
                    train_net.model.save_weights('./checkpoints/{}_train_{}goal_(-{})avg-reward.h5'.format(chosen_training.name, n_goal, np.round(avg_rewards)))
                    break'''

                if (episode%100) == 0:
                    perc_goals = n_goal / 100
                else:
                    perc_goals = n_goal / (episode%100)


                with summary_writer.as_default():
                    tf.summary.scalar('Episode reward', total_reward, step=episode)
                    tf.summary.scalar('Running avg reward (last 100)', avg_rewards, step=episode)
                    tf.summary.scalar('Average loss', loss_mean, step=episode)
                    tf.summary.scalar('Running percentage goals achieved (last 100)', perc_goals, step=episode)

                #Print statistics every 100 cycles
                if episode % 100 == 0:
                    print("EPISODE:", episode, "REWARD:", total_reward, "EPS:", epsilon,
                          "AVG REWARD (LAST 100):", avg_rewards, "LOSS: ", loss_mean, 'N°GOAL REACHED', n_goal)

                    #Condition to save model's weights
                    if avg_rewards > -130:
                        if chosen_training is not TrainingMode.DQN:
                            target_net.model.save_weights('./checkpoints/{}_target_{}goal_(-{})avg-reward.h5'.format(chosen_training.name, n_goal, np.round(avg_rewards)))
                        train_net.model.save_weights('./checkpoints/{}_train_{}goal_(-{})avg-reward.h5'.format(chosen_training.name, n_goal, np.round(avg_rewards)))
                        #break
                    n_goal = 0
        except:
            print('\nTRAIN INTERRUPTED\n')
            print('Please remember to check the statistics with \'tensorboard --logdir train_logs\'')
            sys.exit(1)
                
        print("AVG reward for last 100 episodes:", avg_rewards)
        print('Please remember to check the statistics with \'tensorboard --logdir train_logs\'')
        env.close()
    
    #PLAY MODE
    elif args.mode == "play":
        
        #Plain new model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, activation="relu", input_shape=(num_states,)),
            tf.keras.layers.Dense(48, activation="relu"),
            tf.keras.layers.Dense(num_actions, activation='linear')
        ])
        #Load an already trained model
        
        try:
            print("\nINSERT THE NUMBER OF THE FILE (.h5|.tf) TO LOAD FOR THE PLAY:\n")
            files = sorted([os.path.basename(x) for x in glob.glob('./checkpoints/*.*')])
            for n_file in range(len(files)):
                print('%d: %s' %(n_file + 1, files[n_file]))
            print()
            file_to_load = int(input())
            model.load_weights('./checkpoints/'+files[file_to_load - 1])
        except:
            print("Some problem occured, check if the file exists or if you are in the root of the checkpoints folder!")
            sys.exit(1)

        # Variables for the tensorboard statistics
        current_time = datetime.datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
        log_dir = 'play_logs/' + files[file_to_load -1] + '-' + current_time
        summary_writer = tf.summary.create_file_writer(log_dir)
        
        try:
            n_goal = 0
            #Play 100 times
            print('\nLET\'S PLAY!\n')
            for episode in range(1, 101):
                obs = env.reset()
                rewards = 0     
                for step in range(200):
                    q_values = model.predict(obs[np.newaxis])
                    action = np.argmax(q_values[0])
                    obs, reward, done, _ = env.step(action)
                    rewards += reward
                    env.render()
                    if done:

                        if rewards > -200:
                            n_goal += 1
                            print('Episode %d: Goal achieved in %d steps' %(episode, -rewards))
                        else:
                            print('Episode %d: Goal not achieved' %(episode))

                        with summary_writer.as_default():
                            tf.summary.scalar('Episode reward', rewards, step=episode)
                            tf.summary.scalar('Goals achieved', n_goal, step=episode)

                        break
            print('\nSEE YOU NEXT TIME!\n')
            print('Please remember to check the statistics with \'tensorboard --logdir play_logs\'')
        except:
            print('\nPLAY INTERRUPTED\n')
            print('Please remember to check the statistics with \'tensorboard --logdir play_logs\'')
            
    else:
        print("Error")

if __name__ == '__main__':
    printer.print_title()
    printer.print_subtitle()
    printer.print_car()
    main()
