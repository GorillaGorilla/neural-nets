
import tensorflow as tf
import numpy as np
import cloudpickle as pickle
import gym

learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
D = 80*80

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

class policy:
    def __init__(self, H, D):

        # define forward network
        self.observations = tf.placeholder(tf.float32, [None, D], name="frame_x")
        self.model = model = {}
        model["W1"] = tf.get_variable("W1", shape=[D, H], initializer=tf.contrib.layers.xavier_initializer())
        model["W2"] = tf.get_variable("W2", shape=[H, None], initializer=tf.contrib.layers.xavier_initializer())
        layer1 = tf.nn.relu(tf.matmul(self.observations, model["W1"]))
        self.probability = tf.nn.sigmoid(tf.matmul(layer1, model["W2"]))
        adam = tf.train.AdamOptimizer(learning_rate=learning_rate)



        # training stuff
        self.tvars = tf.trainable_variables()
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.advantages = tf.placeholder(tf.float32, name="reward_signal")
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
        weights = self.sess.run(agent.tvars)
        print(weights)
        return str(weights)

def resetGradBuffer(gradBuffer):
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
    return gradBuffer

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

env = gym.make('Pong-v0')
agent = policy(200, D)
tf.reset_default_graph()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    observation = env.reset()
    gradBuffer = sess.run(agent.tvars)
    gradBuffer = resetGradBuffer(gradBuffer)

    while True:



