import numpy as np
import tensorflow as tf

import gym
env = gym.make('CartPole-v0')

# hyperparameters
H = 8 # number of hidden layer neurons
learning_rate = 1e-2
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?

model_bs = 3 # Batch size when learning from model
real_bs = 3 # Batch size when learning from real environment

# model initialization
D = 4 # input dimensionality

#policy network

class policy_network:
    def __init__(self, H):
        self.observations = tf.placeholder(tf.float32, [None, 4], name="input_x")
        self.W1 = tf.get_variable("W1", shape=[4, H],
                             initializer=tf.contrib.layers.xavier_initializer())
        layer1 = tf.nn.relu(tf.matmul(self.observations, self.W1))
        self.W2 = tf.get_variable("W2", shape=[H, 1],
                             initializer=tf.contrib.layers.xavier_initializer())
        score = tf.matmul(layer1, self.W2)
        self.probability = tf.nn.sigmoid(score)

        self.tvars = tf.trainable_variables()
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.advantages = tf.placeholder(tf.float32, name="reward_signal")
        adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.W1Grad = tf.placeholder(tf.float32, name="batch_grad1")
        self.W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
        batchGrad = [self.W1Grad, self.W2Grad]
        loglik = tf.log(self.input_y * (self.input_y - self.probability) + (1 - self.input_y) * (self.input_y + self.probability))
        loss = -tf.reduce_mean(loglik * self.advantages)
        self.newGrads = tf.gradients(loss, self.tvars)
        self.updateGrads = adam.apply_gradients(zip(batchGrad, self.tvars))

    def trainPolicyNetwork(self, W1Grad, W2Grad):
        sess.run(self.updateGrads, feed_dict={self.W1Grad: W1Grad, self.W2Grad: W2Grad})

    def calculatePolicyGradients(self, epx, epy, discounted_epr):
        newGrads= self.sess.run(self.newGrads, feed_dict={self.observations: epx, self.input_y: epy, self.advantages: discounted_epr})
        return newGrads

    def setSession(self, sess):
        self.sess = sess

    def evaluatePolicy(self, observations):
        tfprob = self.sess.run(self.probability, feed_dict={self.observations: observations})
        return tfprob

    def writeWeights(self):
        weights = self.sess.run(policy_agent.tvars)
        print(weights)
        return str(weights)

# input_data = tf.placeholder(tf.float32, [None, 5])

class model_network:
    def __init__(self):
        mH = 256  # model layer size
        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [mH, 50])
            softmax_b = tf.get_variable("softmax_b", [50])

            self.previous_state = tf.placeholder(tf.float32, [None, 5], name="previous_state")
            W1M = tf.get_variable("W1M", shape=[5, mH],
                                  initializer=tf.contrib.layers.xavier_initializer())
            B1M = tf.Variable(tf.zeros([mH]), name="B1M")
            layer1M = tf.nn.relu(tf.matmul(self.previous_state, W1M) + B1M)
            W2M = tf.get_variable("W2M", shape=[mH, mH],
                                  initializer=tf.contrib.layers.xavier_initializer())
            B2M = tf.Variable(tf.zeros([mH]), name="B2M")
            layer2M = tf.nn.relu(tf.matmul(layer1M, W2M) + B2M)
            wO = tf.get_variable("wO", shape=[mH, 4],
                                 initializer=tf.contrib.layers.xavier_initializer())
            wR = tf.get_variable("wR", shape=[mH, 1],
                                 initializer=tf.contrib.layers.xavier_initializer())
            wD = tf.get_variable("wD", shape=[mH, 1],
                                 initializer=tf.contrib.layers.xavier_initializer())

            bO = tf.Variable(tf.zeros([4]), name="bO")
            bR = tf.Variable(tf.zeros([1]), name="bR")
            bD = tf.Variable(tf.ones([1]), name="bD")

            predicted_observation = tf.matmul(layer2M, wO, name="predicted_observation") + bO
            predicted_reward = tf.matmul(layer2M, wR, name="predicted_reward") + bR
            predicted_done = tf.sigmoid(tf.matmul(layer2M, wD, name="predicted_done") + bD)

            self.true_observation = tf.placeholder(tf.float32, [None, 4], name="true_observation")
            self.true_reward = tf.placeholder(tf.float32, [None, 1], name="true_reward")
            self.true_done = tf.placeholder(tf.float32, [None, 1], name="true_done")

            self.predicted_state = tf.concat([predicted_observation, predicted_reward, predicted_done], 1)

            observation_loss = tf.square(self.true_observation - predicted_observation)

            reward_loss = tf.square(self.true_reward - predicted_reward)

            done_loss = tf.multiply(predicted_done, self.true_done) + tf.multiply(1 - predicted_done, 1 - self.true_done)
            done_loss = -tf.log(done_loss)

            self.model_loss = tf.reduce_mean(observation_loss + done_loss + reward_loss)

            modelAdam = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.updateModel = modelAdam.minimize(self.model_loss)

    def stepModel(self, sess, xs, action):
        toFeed = np.reshape(np.hstack([xs[-1][0], np.array(action)]), [1, 5])
        myPredict = sess.run([self.predicted_state], feed_dict={self.previous_state: toFeed})
        reward = myPredict[0][:, 4]
        observation = myPredict[0][:, 0:4]
        observation[:, 0] = np.clip(observation[:, 0], -2.4, 2.4)
        observation[:, 2] = np.clip(observation[:, 2], -0.4, 0.4)
        doneP = np.clip(myPredict[0][:, 5], 0, 1)
        if doneP > 0.1 or len(xs) >= 300:
            done = True
        else:
            done = False
        return observation, reward, done

    def trainModel(self, epx, epy, epr, epd):
        actions = np.array([np.abs(y - 1) for y in epy][:-1])
        state_prevs = epx[:-1, :]
        state_prevs = np.hstack([state_prevs, actions])
        state_nexts = epx[1:, :]
        rewards = np.array(epr[1:, :])
        dones = np.array(epd[1:, :])
        state_nextsAll = np.hstack([state_nexts, rewards, dones])

        feed_dict = {self.previous_state: state_prevs, self.true_observation: state_nexts, self.true_done: dones,
                     self.true_reward: rewards}
        loss, pState, _ = sess.run([self.model_loss, self.predicted_state, self.updateModel], feed_dict)
        return loss, pState


def resetGradBuffer(gradBuffer):
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
    return gradBuffer


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

xs, drs, ys, ds = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 1
real_episodes = 1
# init = tf.initialize_all_variables()



batch_size = real_bs

drawFromModel = False  # When set to True, will use model for observations
trainTheModel = True  # Whether to train the model
trainThePolicy = False  # Whether to train the policy
switch_point = 1
tf.reset_default_graph()
policy_agent = policy_network(12)
model = model_network()
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:

    policy_agent.setSession(sess)
    rendering = False
    sess.run(init)
    observation = env.reset()
    x = observation
    gradBuffer = sess.run(policy_agent.tvars)
    gradBuffer = resetGradBuffer(gradBuffer)

    while episode_number <= 5000:
        # Start displaying environment once performance is acceptably high.
        if (reward_sum / batch_size > 150 and drawFromModel == False) or rendering == True:
            env.render()
            rendering = True

        x = np.reshape(observation, [1, 4])

        # tfprob = sess.run(probability, feed_dict={observations: x})
        tfprob = policy_agent.evaluatePolicy(x)
        action = 1 if np.random.uniform() < tfprob else 0

        # record various intermediates (needed later for backprop)
        xs.append(x)
        y = 1 if action == 0 else 0
        ys.append(y)

        # step the  model or real environment and get new measurements
        if drawFromModel == False:
            observation, reward, done, info = env.step(action)
        else:
            observation, reward, done = model.stepModel(sess, xs, action)

        reward_sum += reward

        ds.append(done * 1)
        drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

        if done:
            # when the episode is finished deal with training etc
            if drawFromModel == False:
                real_episodes += 1
            episode_number += 1

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            epd = np.vstack(ds)
            xs, drs, ys, ds = [], [], [], []  # reset array memory

            if trainTheModel == True:
                loss, pState = model.trainModel(epx, epy, epr, epd)

            if trainThePolicy == True:
                discounted_epr = discount_rewards(epr).astype('float32')
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)
                tGrad = policy_agent.calculatePolicyGradients(epx, epy, discounted_epr)

                # tGrad = sess.run(newGrads, feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})

                # If gradients becom too large, end training process
                if np.sum(tGrad[0] == tGrad[0]) == 0:
                    break
                for ix, grad in enumerate(tGrad):
                    gradBuffer[ix] += grad

            # change mode if enough episodes have been done in this mode
            if switch_point + batch_size == episode_number:
                switch_point = episode_number
                if trainThePolicy == True:
                    # sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                    policy_agent.trainPolicyNetwork(gradBuffer[0], gradBuffer[1])
                    gradBuffer = resetGradBuffer(gradBuffer)

                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                if drawFromModel == False:
                    print('World Perf: Episode %f. Reward %f. action: %f. mean reward %f.' % (real_episodes, reward_sum / real_bs, action, running_reward / real_bs))
                    if reward_sum / batch_size > 200:
                        # complete
                        policy_agent.writeWeights()
                        file = open('policy_agent_weights.txt', 'w')
                        file.write(policy_agent.writeWeights())
                        file.close()
                        break
                reward_sum = 0

                # Once the model has been trained on 100 episodes, we start alternating between training the policy
                # from the model and training the model from the real environment.
                if episode_number > 100:
                    drawFromModel = not drawFromModel
                    trainTheModel = not trainTheModel
                    trainThePolicy = not trainThePolicy

            if drawFromModel == True:
                observation = np.random.uniform(-0.1, 0.1, [4])  # Generate reasonable starting point
                batch_size = model_bs
            else:
                observation = env.reset()
                batch_size = real_bs






print(real_episodes)