import numpy as np
import cloudpickle as pickle
import gym
import datetime

model = pickle.load(open('save.p', 'rb'))
render = True
D = 80 * 80  # input dimensionality: 80x80 grid
H = 200
prev_x = None  # used in computing the difference frame
print(model);

# resetting env. episode reward total was -16.000000. running mean: -16.691476

def policy_forward(x):
    # print("model['W1'] " , (model['W1']).shape)
    # print("x ", (x.shape))
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    # print("model['W2'] " , (model['W2']).shape)
    # print("h ", (h.shape))
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h  # return probability of taking action 2, and hidden state

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


env = gym.make("Pong-v0")
env = gym.wrappers.Monitor(env, 'pong/' + str(datetime.datetime.now()))
observation = env.reset()
prev_x = None  # used in computing the difference frame
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
while True:
    if render: env.render()
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x
    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

    observation, reward, done, info = env.step(action)
    reward_sum += reward

    if done:  # an episode finished
        break

    if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
        print(('ep %d: game finished, reward: %f' % (1, reward)) + ('' if reward == -1 else ' !!!!!!!!'))