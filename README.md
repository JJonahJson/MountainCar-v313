# MountainCar-v313
### A solution for the MountainCar-v0 problem of the OpenAI Gym environment

An implementation with three different techniques: deep Q learning, fixed Q targets and double deep Q learning. 

Each of them reaches the flag with a percentage of success of 90% very quickly, but only with fixed Q targets and double Q learning the problem is solved as intended by OpenAI Gym’s GitHub site: getting average reward of -110.0 over 100 consecutive trials.

## Results
![](./mountaincar.gif)

Best technique: Double DQN\
The results below come from one training with Double DQN, the csv file of this run are uploaded with the project.

**Solution nearly found at the 816<sup>th</sup> time step** with average reward of 110.8\
**Solution found at the 955<sup>th</sup> time step** with average reward of -109.1\
**Best average reward found is -103.5** at the 1235<sup>th</sup> time step


## How to install
Clone the repository:
```bash
git clone https://github.com/JJonahJson/MountainCar-v313.git
```
Install python3 and pip3
```bash
sudo apt update
sudo apt install python3.7
sudo apt install python3-pip
```

If you have problems installing python3.7, follow these instructions, then try again the steps before:
```
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
```

Install the requirements to launch the bot:
```bash
cd /path/to/MountainCar-v313
pip3 install -r requirements.txt --user
```

## How to launch
There are two principal launch options: **train** and **play**, the first trains your software to solve the MountainCar problem, the second one makes you choose between some of the trained model from the training mode and play the environment with that, be careful to have trained your network at least one time before choosing this mode!

```
cd /path/to/MountainCar-v313/code
python3 main.py -m train
```
The software will ask you which technique you want to train the network.

If you want to try different hyperparameters values from the ones I used:
```
python3 main.py -m train -p
```
The software will ask you all the values you want to set for the hyperparameters.

If you want to see the trials during the training:
```
python3 main.py -m train -s
```
or
```
python3 main.py -m train -p -s
```

If you want to see the behaviour of a saved checkpoint from a training:
```
python3 main.py -m play
```
The software will ask you to choose from all your saved checkpoints.
