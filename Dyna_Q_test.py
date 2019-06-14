import gym
import numpy as np
import time
from Dyna_Q import *

def main():
    env = gym.make('CartPole-v1')
    actions = np.arange(env.action_space.n)
    term_state = -1
    agent = Dyna_Q(actions, term_state, num_sim=0, num_iter=30, method="softmax")
    interval = 250
    cum_rewards = 0
    high = env.observation_space.high

    for i_episode in range(1000):
        observation = env.reset()
        state = generate_state(observation, high)
        for t in range(100):
            env.render()
            action = agent.choose_action(state)
            try:
                observation, reward, done, info = env.step(action)
                state_ = generate_state(observation, high)
                if done:
                    reward = 0
                    state_ = term_state
            except:
                reward = 0
                state_ = term_state
            finally:
                # cum_rewards += reward
                agent.store_result(state, action, reward, state_)
                state = state_
                if reward <= 0 or t == 99:
                    cum_rewards += t
                    agent.update_Models()
                    agent.update_Q()
                    agent.planning()
                    print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                    break
                
        if (i_episode + 1) % interval == 0:
            print("The average reward is : {}".format(cum_rewards / interval))
            cum_rewards = 0

    env.close()


def generate_state(observation, high):
    # Precondition: observation and high are lists of numbers whose lengths are the same
    state = ""
    for obs, h in zip(observation, high):
        negative = 0
        if obs < 0:
            negative = 1
        order = 0
        #if h < 10:
        #    order += 1
        if h < 1:
            s = str(h)
            s_i, s_d = s.split('.')
            order += len(s_d) - len(str(int(s_d))) + 1
        encorded = int(abs(round(obs * 10 ** order, 0))) * 2 + negative
        state += str(encorded)

    return state
    
if __name__ == "__main__":
    main()