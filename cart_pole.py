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
        state = tf.placeholder("float", [None, 4])
        actions = tf.placeholder("float", [None, 2])
        linear = tf.matmul(state, params)

        probabilities = tf.nn.softmax(linear)
        print("probabilities", probabilities)
        good_probabilities = tf.reduce_sum(tf.matmul(probabilities, actions), reduction_indices=[1])
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


# tensorflow operations to compute probabilties for each action, given a state
pl_probabilities, pl_state = policy_gradient()
observation = env.reset()
actions = []
transitions = []
for _ in xrange(200):
    # calculate policy
    obs_vector = np.expand_dims(observation, axis=0)
    probs = sess.run(pl_probabilities,feed_dict={pl_state: obs_vector})
    action = 0 if random.uniform(0,1) < probs[0][0] else 1
    # record the transition
    states.append(observation)
    actionblank = np.zeros(2)
    actionblank[action] = 1
    actions.append(actionblank)
    # take the action in the environment
    old_observation = observation
    observation, reward, done, info = env.step(action)
    transitions.append((old_observation, action, reward))
    totalreward += reward

    if done:
        break

vl_calculated, vl_state, vl_newvals, vl_optimizer = value_gradient()
update_vals = []
for index, trans in enumerate(transitions):
    obs, action, reward = trans
    # calculate discounted monte-carlo return
    future_reward = 0
    future_transitions = len(transitions) - index
    decrease = 1
    for index2 in xrange(future_transitions):
        future_reward += transitions[(index2) + index][2] * decrease
        decrease = decrease * 0.97
    update_vals.append(future_reward)
update_vals_vector = np.expand_dims(update_vals, axis=1)
sess.run(vl_optimizer, feed_dict={vl_state: states, vl_newvals: update_vals_vector})