import numpy as np
############
stateA=0
stateB=1
state_term_neur=2
state_term_rew=3
A_asize=2
B_asize=8
################
#Q learning params
Nruns=100
alpha=0.01
gamma=1
lambd=0.9
tau=100
episodes=5000
evalsteps=100
meanreward=-0.5
epsilon_decay=0.0005
uncertain=1
thresh=0.001
deltab_thresh_max=0.015
deltab_thresh_min=-0.015
deltab_thresh=-0.05
w=1000