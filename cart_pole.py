import numpy as np
import gym
import tensorflow as tf

env = gym.make('CartPole-v0')

observation = env.reset();
parameters = np.random.rand(4) * 2 - 1

print("parameters ", parameters)


def run_episode(env, parameters):
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        env.render()
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward


# random search

def random_search(env, parameters):
    maxReward = 0
    bestParams = parameters
    for _ in range(100):
        parameters = np.random.rand(4) * 2 - 1

        currentReward = run_episode(env, parameters)
        if (currentReward > maxReward):
            bestParams = parameters
            maxReward = currentReward

    print("Best params:")
    print(bestParams)

    print("Best reward")
    print(maxReward)

    run_episode(env, bestParams)

# policy gradient

def policy_gradient():
    with tf.variable_scope("policy"):
        params = tf.get_variable("policy_parameters", [4, 2])
        print("params", params)
        state = tf.placeholder("float", [None, 4])
        actions = tf.placeholder("float", [None, 2])
        linear = tf.matmul(state, params)

        probabilities = tf.nn.softmax(linear)
        print("probabilities", probabilities)
        good_probabilities = tf.reduce_sum(tf.mul(probabilities, actions), reduction_indices=[1])
        print("good_probabilities", good_probabilities)
        # maximize the log probability
        log_probabilities = tf.log(good_probabilities)
        loss = -tf.reduce_sum(log_probabilities)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

def value_gradient():
    with tf.variable_scope("value"):
        state = tf.placeholder("float",[None,4])
        newvals = tf.placeholder("float",[None,1])
        w1 = tf.get_variable("w1",[4,10])
        b1 = tf.get_variable("b1",[10])
        h1 = tf.nn.relu(tf.matmul(state,w1) + b1)
        w2 = tf.get_variable("w2",[10,1])
        b2 = tf.get_variable("b2",[1])
        calculated = tf.matmul(h1,w2) + b2
        diffs = calculated - newvals
        loss = tf.nn.l2_loss(diffs)
        optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
        return calculated, state, newvals, optimizer, loss


policy_gradient()