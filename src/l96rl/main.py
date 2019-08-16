import gym
import matplotlib.pyplot as plt
import numpy as np
import torch as th

##################################################################################
# CREATE THE ENVIRONMENT
##################################################################################

from L96 import L96TwoLevel

class L96TwoLevelRL(L96TwoLevel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step_with_B(self, B):
        k1_X = self._rhs_X_dt(self.X, B=B)
        k2_X = self._rhs_X_dt(self.X + k1_X / 2, B=B)
        k3_X = self._rhs_X_dt(self.X + k2_X / 2, B=B)
        k4_X = self._rhs_X_dt(self.X + k3_X, B=B)

        self.X += 1 / 6 * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)
        #         self.X += B * self.dt

        self.step_count += 1
        if self.step_count % self.save_steps == 0:
            Y_mean = self.Y.reshape(self.K, self.J).mean(1)
            Y2_mean = (self.Y.reshape(self.K, self.J) ** 2).mean(1)
            self._history_X.append(self.X.copy())
            self._history_Y_mean.append(Y_mean.copy())
            self._history_Y2_mean.append(Y2_mean.copy())
            self._history_B.append(B.copy())
            if not self.noYhist:
                self._history_Y.append(self.Y.copy())

class L96Gym(gym.Env):
    def __init__(self, lead_time=1., dt=0.01, action_bounds=(-20, 20)):
        """Lorenz 96 gym environment.

        Every reset a new random state from the history file is chosen for initial
        conditions. The reward is zero for all steps until the lead time is reached.
        Then the final reward is the negative MSE to the "true" model.

        Both state and action of size K(=36 by default) x 1. The trailing dimension
        is for convenience, since many ML frameworks require a sample dimension.

        Args:
          history: reference forecast history from which to draw initial conditions
                   and the "true" forecast
          lead_time: forecast lead time in L96 model time units
          dt: Time step of parameterized model
          action_bounds: clip predicted actions at these bounds
        """
        self.lead_time, self.dt = lead_time, dt
        self.nsteps = self.lead_time // self.dt
        self.step_count = 0

        from gym import spaces
        self.action_space = spaces.Box(
            low=np.array([action_bounds[0]]),
            high=np.array([action_bounds[1]])
        )

        # do I really need an infinite observation space? Should I maybe tanh the observations or such?
        self.observation_space = gym.spaces.Box(-np.array([np.inf]), np.array([np.inf]))

        # load dataset
        # try:
        import xarray as xr
        self.h = xr.open_dataset('../data/L96TwoLevel_ref.nc')
        # except:
        #     from data.import_npy import NPYDataset
        #     self.h = NPYDataset()


    def reset(self):
        self.init_time = np.random.choice(self.h.time[:-int(self.lead_time / 0.1)])
        self.X_init = self.h.X.sel(time=self.init_time, method='nearest').values
        self.fc_target = self.h.X.sel(time=self.init_time + self.lead_time, method='nearest').values

        self.l96 = L96TwoLevelRL(noYhist=True, X_init=self.X_init, dt=self.dt, save_dt=self.dt)
        state = self.l96.X.copy()
        self.step_count = 0
        return state[:, None]

    def step(self, action):
        self.l96.step_with_B(action.squeeze())
        state = self.l96.X.copy()
        self.step_count += 1
        if self.step_count >= self.nsteps:
            done = True
            reward = -((state - self.fc_target) ** 2).mean()
        else:
            done = False
            reward = 0
        return state[:, None], reward, done, None

##################################################################################
# CREATE REPLAY BUFFER
##################################################################################

from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        if state.ndim == 1:
            self.buffer.append([state, action, reward, next_state, done])
        else:
            for i in range(len(state)):
                self.buffer.append([
                    state[i], action[i], reward, next_state[i], done])

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

##################################################################################
# CREATE LEARNING ALGORITHM - DDPG
##################################################################################

def ddpg_update(batch_size,
                gamma=0.99,
                min_value=-np.inf,
                max_value=np.inf,
                soft_tau=1e-2):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = th.FloatTensor(state).to(device)
    next_state = th.FloatTensor(next_state).to(device)
    action = th.FloatTensor(action).to(device)
    reward = th.FloatTensor(reward).unsqueeze(1).to(device)
    done = th.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    policy_loss = value_net(state, policy_net(state))
    policy_loss = -policy_loss.mean()

    next_action = target_policy_net(next_state)
    target_value = target_value_net(next_state, next_action.detach())
    # Episodes should be continuous - i.e. non-terminating.
    # Otherwise the network has to learn what states are terminal - but those
    # are in no way different from those which aren't!
    done = 0.0 # DEBUG
    expected_value = reward + (1.0 - done) * gamma * target_value
    expected_value = th.clamp(expected_value, min_value, max_value)

    value = value_net(state, action)
    value_loss = value_criterion(value, expected_value.detach())

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )

    return policy_loss, value, target_value, value_loss

##################################################################################
# CREATE NETWORK MODELS
##################################################################################
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

class CentralVCritic(nn.Module):
    def __init__(self, scheme, args):
        super(CentralVCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, 64)
        # self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        x = F.relu(self.fc1(inputs))
        # x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = th.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetworkLinear(nn.Module):
    def __init__(self, num_inputs, num_actions, init_w=3e-3):
        super(PolicyNetworkLinear, self).__init__()

        self.linear1 = nn.Linear(num_inputs, num_actions, bias=True)
        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear1.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        return self.linear1(state)

    def get_action(self, state):
        state = np.atleast_2d(state)
        state = torch.FloatTensor(state).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy().squeeze()

class PolicyNetworkNN(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(PolicyNetworkNN, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def get_action(self, state):
        state = np.atleast_2d(state)
        state = th.FloatTensor(state).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy().squeeze()

##################################################################################
# CREATE TRAINING LOOP
##################################################################################

device = "cpu"
env = L96Gym(lead_time=1)
features = np.ravel(env.h.X.values)
targets = np.ravel(env.h.B.values)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
hidden_dim = 256

value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net = PolicyNetworkNN(state_dim, action_dim, hidden_dim).to(device)

target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
target_policy_net = PolicyNetworkNN(state_dim, action_dim, hidden_dim).to(device)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)

for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
    target_param.data.copy_(param.data)

value_lr = 1e-3
policy_lr = 1e-4

from torch import optim
value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)

value_criterion = nn.MSELoss()

replay_buffer_size = 1000000
replay_buffer = ReplayBuffer(replay_buffer_size)

max_frames = 100_000
max_steps = 500
frame_idx = 0
rewards = []
batch_size = 128
weights = []
biases = []

##################################################################################
# PLOTTING ROUTINES
##################################################################################

# Plotting function to track progress
def plot(episode,
         rewards,
         policy_losses,
         values,
         target_values,
         value_losses,
         weights,
         biases):
    #clear_output(True)
    fig, axs = plt.subplots(3, 2, figsize=(10,5))
    axs[0,0].set_title('frame %s. reward: %s' % (episode, np.mean(rewards[-10:])))
    axs[0,0].plot(rewards)
    axs[0,1].scatter(features[::1000], targets[::1000], s=5, alpha=0.2)
    a = np.linspace(-10,15)
    b = policy_net.get_action(a[:, None])
    axs[0, 1].plot(a, b, c='orange')
    axs[1, 0].set_title('frame %s. policy loss: %s' % (episode, np.mean(policy_losses[-10:])))
    axs[1, 0].plot(policy_losses)
    axs[1, 1].set_title('frame %s. value loss: %s' % (episode, np.mean(value_losses[-10:])))
    axs[1, 1].plot(value_losses)
    axs[2, 0].set_title('frame %s. value: %s' % (episode, np.mean(values[-10:])))
    axs[2, 0].plot(policy_losses)
    axs[2, 1].set_title('frame %s. target_values: %s' % (episode, np.mean(target_values[-10:])))
    axs[2, 1].plot(value_losses)
    plt.show()

##################################################################################
# START TRAINING
##################################################################################

print("STARTING TRAINING LOOP")
policy_losses = []
values = []
target_values = []
value_losses = []

episode = 0
while frame_idx < max_frames:
    state = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        action = policy_net.get_action(state)
        # I implement exploration noise like this
        # action = action * noise_slope + noise_bias
        noise_slope = np.random.normal(1, 0.1)
        noise_bias = np.random.normal(0, 0.1)
        action = action * noise_slope + noise_bias
        next_state, reward, done, _ = env.step(action)

        replay_buffer.push(state, action[:, None], reward, next_state, done)
        if len(replay_buffer) > batch_size:
            policy_loss, value, target_value, value_loss = ddpg_update(batch_size)
            policy_losses.append(policy_loss)
            values.append(value)
            target_values.append(target_value)
            value_losses.append(value_loss)

        state = next_state
        episode_reward += reward
        frame_idx += 1

        if frame_idx % max(1000, max_steps + 1) == 0:
            plot(frame_idx,
                 rewards,
                 [p.numpy() for p in policy_losses],
                 [p.numpy() for p in values],
                 [p.numpy() for p in target_values],
                 [p.numpy() for p in value_losses],
                 weights,
                 biases)

        if done:
            weights.append(policy_net.linear1.weight.cpu().detach().numpy()[0].copy())
            biases.append(policy_net.linear1.bias.cpu().detach().numpy()[0].copy())
            episode += 1
            break

    rewards.append(episode_reward)


