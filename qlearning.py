import numpy as np
import matplotlib.pyplot as plt

from statistics import stdev, mean

import gym



def plot(rewards):
    plt.plot(rewards)
    plt.show()


def fourier_basis(obs, order=3):
    # use fourier basis to expand state and create coefficients for linear function approximator
    obs = np.array([(obs[0]+4.8)/9, (obs[1]+100)/200, (obs[2]+24)/48, (obs[3]+100)/200])
    return np.sum([np.cos(i*np.pi*obs) for i in range(1, order+1)], axis=0)


def get_q(weights, obs, action, order=3):
    # expand state representation and combine with weights to give values for given action
    features = fourier_basis(obs, order) * weights[action*obs.size:(action+1)*obs.size]
    # concatenate with zeros so that q_values van be added to weights for correct action in update
    return np.concatenate([features if i == action else np.zeros(features.size) for i in range(weights.size//obs.size)])    


def max_action(weights, obs, order=3):
    # find q-values for each state
    q_values = [np.sum(get_q(weights, obs, action, order)) for action in range(weights.size//obs.size)]
    # choose action based on which has the larger q-value
    return np.argmax(q_values)


def epsilon_greedy(episode, weights, obs, order=3, epsilon=0.3):
    # with probability 1-epsilon, choose action greedily (i.e. action with max probability)
    if np.random.rand() > epsilon**episode:
        return max_action(weights, obs, order)
    # with probability epsilon, choose action at random
    else:
        return np.random.choice(weights.size//obs.size)


def train(env, n_episodes=500, max_steps=1000):
    print('Start Training...')

    order = 5           # order of fourier expansion for function approximation
    epsilon = 0.3       # initial epsilon for policy
    gamma = 0.9         # reward discount
    alpha = 0.1         # learning rate

    # initialize random weights close to zero with size set so that each action can be represented 
    # by a vector with the size of the observation times weights
    weights = np.random.normal(loc=0, scale=0.5, 
                size=env.observation_space.shape[0]*env.action_space.n)

    avg_returns = []

    for episode in range(1, n_episodes+1):
        rewards = []

        # draw new state and reset environment
        obs = env.reset()

        for ts in range(max_steps):
            # get action from epsilon greedy policy
            action = epsilon_greedy(episode, weights, obs, order, epsilon)

            # get new observation using current action
            next_obs, reward, done, _ = env.step(action)

            # calculate q(next_obs, max action)
            qnext = get_q(weights, next_obs, max_action(weights, next_obs))

            # calculate q(obs,action)
            q = get_q(weights, obs, action)

            # calculate td-error
            td_error = reward + gamma*qnext - q

            # calculate weight q-learning update
            weights += alpha * td_error

            if done:
                break

            # make next observation current observation for next timestep
            obs = next_obs

            rewards.append(reward)

        if episode % 100 == 0:
            alpha = alpha // 2

        avg_returns.append(sum(rewards))

        if episode % 10 == 0:
            print('Episode: {} - Episode Return: {:.2f} - Average Return: {:.2f}'.format(
                  episode, sum(rewards), mean(avg_returns)))

    plot(avg_returns)


def seed(env, seed=0):
    env.seed(0)
    np.random.seed(0)


def main(n_episodes=500, max_steps=1000):
    env = gym.make('CartPole-v0')
    seed(env)
    
    train(env, n_episodes, max_steps)


if __name__ == '__main__':
    main()