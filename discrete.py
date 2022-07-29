import collections, gym
import numpy as np
import matplotlib.pyplot as plt
# import import_ipynb
# import lunar_lander as lander


def discrete_states(observation):

    ds_vector = np.zeros(6)
    ds=-1
    dis_number = 10


    #Discretize x
    if observation[0] <= -1:
        ds_vector[0] = -1
    elif observation[0] >= 1:
        ds_vector[0] = 1
    else:
        for i in range(dis_number):
            if -1 <= observation[0] <= (-1 + i*2/dis_number):
                ds_vector[0] = -1 + i*2/dis_number
                break


    #Discretize y
    if observation[1] <= -1:
        ds_vector[1] = -1
    elif observation[1] >= 1:
        ds_vector[1] = 1

    else:
        for i in range(dis_number):
            if -1 <= observation[1] <= (-1 + i*2/dis_number):
                ds_vector[1] = -1 + i*2/dis_number
                break


    #Discretize Vx
    if observation[2] <= -1.5:
        ds_vector[2] = -1.5
    elif observation[2] >= 1.5:
        ds_vector[2] = 1.5

    else:
        for i in range(dis_number):
            if -1.5 <= observation[2] <= (-1.5 + i*3/dis_number):
                ds_vector[2] = -1.5 + i*3/dis_number
                break


    #Discretize Vy
    if observation[3] <= -1.5:
        ds_vector[3] = -1.5
    elif observation[3] >= 1.5:
        ds_vector[3] = 1.5

    else:
        for i in range(dis_number):
            if -1.5 <= observation[3] <= (-1.5 + i*3/dis_number):
                ds_vector[3] = -1.5 + i*3/dis_number
                break

    #Discretize theta
    if observation[4] <= -2:
        ds_vector[4] = -2
    elif observation[4] >= 2:
        ds_vector[4] = 2

    else:
        for i in range(dis_number):
            if -2 <= observation[4] <= (-2 + i*4/dis_number):
                ds_vector[4] = -2 + i*4/dis_number
                break


    #Discretize V_theta
    if observation[5] <= -6:
        ds_vector[5] = -6
    elif observation[5] >= 6:
        ds_vector[5] = 6

    else:
        for i in range(dis_number):
            if -6 <= observation[5] <= (-6 + i*12/dis_number):
                ds_vector[5] = -6 + i*12/dis_number
                break

    return ds_vector



def sa_key(s, a):
    return str(s) + " " + str(a)


def policy_explorer(s, Q, iter):
    rand = np.random.randint(0, 100)

    threshold = 20

    if rand >= threshold:
        Qv = np.array([ Q[sa_key(s, action)] for action in [0, 1, 2, 3]])
        return np.argmax(Qv)
    else:
        return np.random.randint(0, 4)




def sarsa_lander(env, seed=None, render=False, num_iter=50, seg=50):
    env.seed(42)

    Q = collections.defaultdict(float)

    gamma = 0.95

    r_seq = []
    it_reward = []

    for it in range(num_iter):
        # initialize variables
        total_reward = 0
        steps = 0

        lr = 1e-2

        # reset environment
        s = env.reset()

        ds = discrete_states(s)
        a = policy_explorer(ds, Q, it)
        # start Sarsa
        while True:
            sa = sa_key(ds, a)
            if render:
                env.render()
            sp, r, done, info = env.step(a)
            # update corresponding Q
            dsp = discrete_states(sp)
            ap = policy_explorer(dsp, Q, it)
            next_sa = sa_key(dsp, ap)
            if not done:
                Q[sa] += lr*(r + gamma * Q[next_sa] - Q[sa])
            else:
                Q[sa] += lr*(r - Q[sa])
            ds = dsp
            a = ap
            total_reward += r
            steps += 1
            if done or steps > 1000:
                it_reward.append(total_reward)
                break
        if it % seg == 0:
            avg_rwd = np.mean(np.array(it_reward))
            print("#It: ", it, " avg reward: ", avg_rwd, " out of ", len(it_reward), " trials")
            it_reward = []
            r_seq.append(avg_rwd)

    return Q, r_seq

num_iter = 4000

env = gym.make("LunarLander-v2")
Q, r_seq = sarsa_lander(env, render=True, num_iter=num_iter, seg=100)


y = np.array(r_seq)
x = np.linspace(0, num_iter, y.shape[0])
plt.figure(figsize=[10,10])
plt.plot(x, y)
plt.title('10 value discretized SARSA')
plt.savefig("10 value discretized SARSA")
plt.close()

np.savetxt("10 v sarsa.txt", y)
