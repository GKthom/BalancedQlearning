#sample call:
#python balanced.py -g 'game' -v
#'game' can be breakout/space_invaders/asterix/seaquest or freeway
##############################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import time

import random, numpy, argparse, logging, os

from collections import namedtuple
from minatar import Environment

################################################################################################################
# Constants
#
################################################################################################################
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 100000
TARGET_NETWORK_UPDATE_FREQ = 1000
TRAINING_FREQ = 1
NUM_FRAMES = 5000000
FIRST_N_FRAMES = 100000
REPLAY_START_SIZE = 5000
END_EPSILON = 0.1
STEP_SIZE = 0.00025
GRAD_MOMENTUM = 0.95
SQUARED_GRAD_MOMENTUM = 0.95
MIN_SQUARED_GRAD = 0.01
GAMMA = 0.99
EPSILON = 1.0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


################################################################################################################
# class QNetwork
#
# One hidden 2D conv with variable number of input channels.  We use 16 filters, a quarter of the original DQN
# paper of 64.  One hidden fully connected linear layer with a quarter of the original DQN paper of 512
# rectified units.  Finally, the output layer is a fully connected linear layer with a single output for each
# valid action.
#
################################################################################################################
class QNetwork(nn.Module):
    def __init__(self, in_channels, num_actions):

        super(QNetwork, self).__init__()

        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)

        # Output layer:
        self.output = nn.Linear(in_features=128, out_features=num_actions)

    # As per implementation instructions according to pytorch, the forward function should be overwritten by all
    # subclasses
    def forward(self, x):
        # Rectified output from the first conv layer
        x = f.relu(self.conv(x))

        # Rectified output from the final hidden layer
        x = f.relu(self.fc_hidden(x.view(x.size(0), -1)))

        # Returns the output from the fully-connected linear layer
        return self.output(x)


###########################################################################################################
# class replay_buffer
#
# A cyclic buffer of a fixed size containing the last N number of recent transitions.  A transition is a
# tuple of state, next_state, action, reward, is_terminal.  The boolean is_terminal is used to indicate
# whether if the next state is a terminal state or not.
#
###########################################################################################################
transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')
class replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []

    def add(self, *args):
        # Append when the buffer is not full but overwrite when the buffer is full
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*args))
        else:
            self.buffer[self.location] = transition(*args)

        # Increment the buffer location
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


################################################################################################################
# get_state
#
# Converts the state given by the environment to a tensor of size (in_channel, 10, 10), and then
# unsqueeze to expand along the 0th dimension so the function returns a tensor of size (1, in_channel, 10, 10).
#
# Input:
#   s: current state as numpy array
#
# Output: current state as tensor, permuted to match expected dimensions
#
################################################################################################################
def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()


################################################################################################################
# world_dynamics
#
# It generates the next state and reward after taking an action according to the behavior policy.  The behavior
# policy is epsilon greedy: epsilon probability of selecting a random action and 1 - epsilon probability of
# selecting the action with max Q-value.
#
# Inputs:
#   t : frame
#   replay_start_size: number of frames before learning starts
#   num_actions: number of actions
#   s: current state
#   env: environment of the game
#   policy_net: policy network, an instance of QNetwork
#
# Output: next state, action, reward, is_terminated
#
################################################################################################################
def world_dynamics(t, replay_start_size, num_actions, s, env, policy_net):

    # A uniform random policy is run before the learning starts
    if t < replay_start_size:
        action = torch.tensor([[random.randrange(num_actions)]], device=device)
    else:
        # Epsilon-greedy behavior policy for action selection
        # Epsilon is annealed linearly from 1.0 to END_EPSILON over the FIRST_N_FRAMES and stays 0.1 for the
        # remaining frames
        epsilon = END_EPSILON if t - replay_start_size >= FIRST_N_FRAMES \
            else ((END_EPSILON - EPSILON) / FIRST_N_FRAMES) * (t - replay_start_size) + EPSILON

        if numpy.random.binomial(1, epsilon) == 1:
            action = torch.tensor([[random.randrange(num_actions)]], device=device)
        else:
            # State is 10x10xchannel, max(1)[1] gives the max action value (i.e., max_{a} Q(s, a)).
            # view(1,1) shapes the tensor to be the right form (e.g. tensor([[0]])) without copying the
            # underlying tensor.  torch._no_grad() avoids tracking history in autograd.
            with torch.no_grad():
                action = policy_net(s).max(1)[1].view(1, 1)

    # Act according to the action and observe the transition and reward
    reward, terminated = env.act(action)

    # Obtain s_prime
    s_prime = get_state(env.state())

    return s_prime, action, torch.tensor([[reward]], device=device).float(), torch.tensor([[terminated]], device=device)


################################################################################################################
# train_balanced
# Inputs:
#   sample: a batch of size 1 or 32 transitions
#   policy_net: an instance of QNetwork
#   target_net: an instance of QNetwork
#   optimizer: centered RMSProp
#
################################################################################################################

def train_balanced(sample, policy_net, target_net, optimizer,old_targ_net,old_policy_net,t):
    # Batch is a list of namedtuple's, the following operation returns samples grouped by keys
    batch_samples = transition(*zip(*sample))

    # states, next_states are of tensor (BATCH_SIZE, in_channel, 10, 10) - inline with pytorch NCHW format
    # actions, rewards, is_terminal are of tensor (BATCH_SIZE, 1)
    states = torch.cat(batch_samples.state)
    next_states = torch.cat(batch_samples.next_state)
    actions = torch.cat(batch_samples.action)
    rewards = torch.cat(batch_samples.reward)
    is_terminal = torch.cat(batch_samples.is_terminal)

    # Obtain a batch of Q(S_t, A_t) and compute the forward pass.
    # Q_s_a is of size (BATCH_SIZE, 1).
    Q_s_a = policy_net(states).gather(1, actions)
    # Get the indices of next_states that are not terminal
    none_terminal_next_state_index = torch.tensor([i for i, is_term in enumerate(is_terminal) if is_term == 0], dtype=torch.int64, device=device)
    # Select the indices of each row
    none_terminal_next_states = next_states.index_select(0, none_terminal_next_state_index)

    Q_s_prime_a_prime_max = torch.zeros(len(sample), 1, device=device)
    Q_s_prime_a_prime_min = torch.zeros(len(sample), 1, device=device)
    if len(none_terminal_next_states) != 0:
        #compute max terms
        Q_s_prime_a_prime_max[none_terminal_next_state_index] = target_net(none_terminal_next_states).detach().max(1)[0].unsqueeze(1)
        #compute min terms
        Q_s_prime_a_prime_min[none_terminal_next_state_index] = target_net(none_terminal_next_states).detach().min(1)[0].unsqueeze(1)
    #use max and min terms to find beta
    beta=findbeta(old_targ_net,old_policy_net,target_net,policy_net,states,actions,rewards,next_states)
    # Compute the balanced target
    target = rewards + GAMMA * ((beta* Q_s_prime_a_prime_max)+((1-beta)* Q_s_prime_a_prime_min))
    # Huber loss
    #loss = f.smooth_l1_loss(target, Qnorm)
    loss = f.smooth_l1_loss(target, Q_s_a)

    # Zero gradients, backprop, update the weights of policy_net
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


#################################################################################################################
# findbeta
#   Inputs:
#   sample: a batch of transitions
#   policy_net: an instance of QNetwork
#   target_net: an instance of QNetwork
#   old_policy_net: previous instance of policy network
#   old_targ_net: previous instance of target network
#   states,actions,rewards and next_states from the batch
#
################################################################################################################
def findbeta(old_targ_net,old_policy_net,target_net,policy_net,states,actions,rewards,next_states):
    #Setting beta_n and beta_n_1 to compute td_n(\delta_n) and td_n_1(\delta_{n+1}) 
    beta_n=1
    beta_n_1=1
    #max limit for beta
    betamax=1
    #min limit for beta
    betamin=0
    #compute min and max targets as per previous target network
    oldQ_s_prime_a_prime_max = old_targ_net(next_states).detach().max(1)[0].unsqueeze(1)
    oldQ_s_prime_a_prime_min = old_targ_net(next_states).detach().min(1)[0].unsqueeze(1)
    #compute min and max targets as per current target network
    newQ_s_prime_a_prime_max = target_net(next_states).detach().max(1)[0].unsqueeze(1)
    newQ_s_prime_a_prime_min = target_net(next_states).detach().min(1)[0].unsqueeze(1)
    #set eta value
    eta=0.2
    inds=actions.long()
    #Q values as per current policy network
    Qval=policy_net(states).gather(1,inds.view(-1,1))
    #Q values as per previous policy network
    Qvalold=old_policy_net(states).gather(1,inds.view(-1,1))
    #TD error as per new policy and target networks
    td_n_1=rewards+GAMMA*((beta_n_1*newQ_s_prime_a_prime_max)+(1-beta_n_1)*newQ_s_prime_a_prime_min)-Qval
    #TD error as per previous policy and target networks
    td_n=rewards+GAMMA*((beta_n*oldQ_s_prime_a_prime_max)+(1-beta_n)*oldQ_s_prime_a_prime_min)-Qvalold
    #Modified TD error
    deltadashnew=td_n_1+(eta*td_n)
    #Compute beta values for each transition in the batch
    beta_all=(deltadashnew-rewards+Qval-GAMMA*newQ_s_prime_a_prime_min)/(newQ_s_prime_a_prime_max-newQ_s_prime_a_prime_min)/GAMMA
    #Clip beta values to the range [0,1]
    beta_all=torch.clamp(beta_all,min=betamin,max=betamax)
    #Compute mean of beta for the batch
    beta_batch=torch.mean(beta_all)

    return beta_batch


################################################################################################################
# balanced dqn
#
# Balanced DQN algorithm 
#
# Inputs:
#   env: environment of the game
#   replay_off: disable the replay buffer and train on each state transition
#   target_off: disable target network
#   output_file_name: directory and file name prefix to output data and network weights, file saved as 
#       <output_file_name>_data_and_weights
#   store_intermediate_result: a boolean, if set to true will store checkpoint data every 1000 episodes
#       to a file named <output_file_name>_checkpoint
#   load_path: file path for a checkpoint to load, and continue training from
#   step_size: step-size for RMSProp optimizer

def balanced_dqn(run_no, env, replay_off, target_off, output_file_name, store_intermediate_result=False, load_path=None, step_size=STEP_SIZE):
    print(run_no)
    # Get channels and number of actions specific to each game
    in_channels = env.state_shape()[2]
    num_actions = env.num_actions()
    # Instantiate networks, optimizer, loss and buffer
    old_targ_net = QNetwork(in_channels, num_actions).to(device)
    old_policy_net = QNetwork(in_channels, num_actions).to(device)
    #Copy of policy and target networks
    polcopy = QNetwork(in_channels, num_actions).to(device)
    targcopy = QNetwork(in_channels, num_actions).to(device)
    policy_net = QNetwork(in_channels, num_actions).to(device)
    replay_start_size = 0
    if not target_off:
        target_net = QNetwork(in_channels, num_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())

    if not replay_off:
        r_buffer = replay_buffer(REPLAY_BUFFER_SIZE)
        replay_start_size = REPLAY_START_SIZE

    optimizer = optim.RMSprop(policy_net.parameters(), lr=step_size, alpha=SQUARED_GRAD_MOMENTUM, centered=True, eps=MIN_SQUARED_GRAD)
    # Set initial values
    e_init = 0
    t_init = 0
    policy_net_update_counter_init = 0
    avg_return_init = 0.0
    data_return_init = []
    frame_stamp_init = []

    # Load model and optimizer if load_path is not None
    if load_path is not None and isinstance(load_path, str):
        checkpoint = torch.load(load_path)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

        if not target_off:
            target_net.load_state_dict(checkpoint['target_net_state_dict'])

        if not replay_off:
            r_buffer = checkpoint['replay_buffer']

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        e_init = checkpoint['episode']
        t_init = checkpoint['frame']
        policy_net_update_counter_init = checkpoint['policy_net_update_counter']
        avg_return_init = checkpoint['avg_return']
        data_return_init = checkpoint['return_per_run']
        frame_stamp_init = checkpoint['frame_stamp_per_run']

        # Set to training mode
        policy_net.train()
        if not target_off:
            target_net.train()

    # Data containers for performance measure and model related data
    data_return = data_return_init
    frame_stamp = frame_stamp_init
    avg_return = avg_return_init

    # Train for a number of frames
    t = t_init
    e = e_init
    policy_net_update_counter = policy_net_update_counter_init
    t_start = time.time()
    reward_seq=[]
    while t < NUM_FRAMES:
        # Initialize the return for every episode (we should see this eventually increase)
        G = 0.0
        # Initialize the environment and start state
        env.reset()
        s = get_state(env.state())
        is_terminated = False
        while(not is_terminated) and t < NUM_FRAMES:
            #print(t)
            if t%100000==0:
                print(t)
            # Generate data
            s_prime, action, reward, is_terminated = world_dynamics(t, replay_start_size, num_actions, s, env, policy_net)

            sample = None
            if replay_off:
                sample = [transition(s, s_prime, action, reward, is_terminated)]
            else:
                # Write the current frame to replay buffer
                r_buffer.add(s, s_prime, action, reward, is_terminated)

                # Start learning when there's enough data and when we can sample a batch of size BATCH_SIZE
                if t > REPLAY_START_SIZE and len(r_buffer.buffer) >= BATCH_SIZE:
                    # Sample a batch
                    sample = r_buffer.sample(BATCH_SIZE)
            #copy policy and target networks
            polcopy.load_state_dict(policy_net.state_dict())
            targcopy.load_state_dict(policy_net.state_dict())
            # Train every n number of frames defined by TRAINING_FREQ
            if t % TRAINING_FREQ == 0 and sample is not None:
                if target_off:
                    train_balanced(sample, policy_net, policy_net, optimizer, old_targ_net,old_policy_net,t)
                else:
                    policy_net_update_counter += 1
                    train_balanced(sample, policy_net, target_net, optimizer, old_targ_net,old_policy_net,t)

            # Update the target network only after some number of policy network updates
            if not target_off and policy_net_update_counter > 0 and policy_net_update_counter % TARGET_NETWORK_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())
            #update previous copies of target and policy networks
            old_targ_net.load_state_dict(targcopy.state_dict())
            old_policy_net.load_state_dict(polcopy.state_dict())

            G += reward.item()
            reward_seq.append(reward.item())

            t += 1

            # Continue the process
            s = s_prime

        # Increment the episodes
        e += 1
        #print(e)

        # Save the return for each episode
        data_return.append(G)
        frame_stamp.append(t)

        # Logging exponentiated return only when verbose is turned on and only at 1000 episode intervals
        avg_return = 0.99 * avg_return + 0.01 * G
        if e % 1000 == 0:
            logging.info("Episode " + str(e) + " | Return: " + str(G) + " | Avg return: " +
                         str(numpy.around(avg_return, 2)) + " | Frame: " + str(t)+" | Time per frame: " +str((time.time()-t_start)/t) )

        # Save model data and other intermediate data if the corresponding flag is true
        if store_intermediate_result and e % 1000 == 0:
            torch.save({
                        'reward_seq': reward_seq,
                        'episode': e,
                        'frame': t,
                        'policy_net_update_counter': policy_net_update_counter,
                        'policy_net_state_dict': policy_net.state_dict(),
                        'target_net_state_dict': target_net.state_dict() if not target_off else [],
                        'optimizer_state_dict': optimizer.state_dict(),
                        'avg_return': avg_return,
                        'return_per_run': data_return,
                        'frame_stamp_per_run': frame_stamp,
                        'replay_buffer': r_buffer if not replay_off else []
            }, output_file_name + "_checkpoint")

    # Print final logging info
    logging.info("Avg return: " + str(numpy.around(avg_return, 2)) + " | Time per frame: " + str((time.time()-t_start)/t))
    # Write data to file
    torch.save({
        'reward_seq': reward_seq,
        'returns': data_return,
        'frame_stamps': frame_stamp,
        'policy_net_state_dict': policy_net.state_dict()
    }, output_file_name + str(run_no)+"_data_and_weights_balanced")


def main(run_no):
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", "-g", type=str)
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--loadfile", "-l", type=str)
    parser.add_argument("--alpha", "-a", type=float, default=STEP_SIZE)
    parser.add_argument("--save", "-s", action="store_true")
    parser.add_argument("--replayoff", "-r", action="store_true")
    parser.add_argument("--targetoff", "-t", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # If there's an output specified, then use the user specified output.  Otherwise, create file in the current
    # directory with the game's name.
    if args.output:
        file_name = args.output
    else:
        file_name = os.getcwd() + "/" + args.game

    load_file_path = None
    if args.loadfile:
        load_file_path = args.loadfile

    env = Environment(args.game)

    print('Cuda available?: ' + str(torch.cuda.is_available()))
    balanced_dqn(run_no, env, args.replayoff, args.targetoff, file_name, args.save, load_file_path, args.alpha)


if __name__ == '__main__':
    Nruns=10
    for i in range(Nruns):
        main(i)