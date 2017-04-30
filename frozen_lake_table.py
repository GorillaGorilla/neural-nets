import gym
import numpy as np

env = gym.make('FrozenLake-v0')

#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
print("Q", Q)
# Set learning parameters
lr = .85
y = .99
num_episodes = 500
#create lists to contain total rewards and steps per episode
#jList = []
rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        j+=1
        #Choose an action by greedily (with noise) picking from Q table
        print(i, j,"Q[s,:]", Q[s,:], s)
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        print(i, j,"a", a)
        #Get new state and reward from environment
        s1,r,d,_ = env.step(a)
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        print(i, j, "CurrentQ")
        print(Q)
        rAll += r
        s = s1
        if d == True:
            if r == 0:
                print("Died")
            else:
                print("Succeeded")
            break
    #jList.append(j)
    rList.append(rAll)

print("Score over time: ",  str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)