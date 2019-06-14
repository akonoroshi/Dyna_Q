# Dyna_Q

This is an implementation of Dyna-Q reinforcement learning algorithm, which is a variant of Q-learning. One of the problems is Q-learning is that we have to interact with the real world in order to update q values, but this process can be costly because the agent chooses sub-optimal actions during exploration. Dyna-Q tries to solve this through planning where the agant do simulations with data it currently has.

## How to use
To initialize Dyna-Q agent, you will need a list of available actions and a terminal state: 

```
agent = Dyna_Q(list_of_actions, terminal_state)
```

Other parameters are pre-determined but feel free to change them to optimize for your environment and/or goal. Note that Dyna_Q.py only supports discrete states and cannot take a list as a state. Please disctetize and concatenate if necessary.

You can also use this as a simple Q-learning agent by setting `num_sim=0`.

Two exploration methods are implemented here; one is Upper-Confidence-Bound (UCB) and softmax. They are specified by `method` parameter.

When you want to take actions based on data, simply call `agent.choose_action(observation)`. This will return an optimal action. Once you see reward and new observation, store the data by calling `agent.store_result(old_observation, action, reward, new_observation)`.

After you finish one episode, call
```
agent.update_Models()
agent.update_Q()
agent.planning()
```
in this order. Otherwise, you cannot Q-values properly. Now, you are free to go to the next episode!

## Known issues (please help me!)
In softmax exploration method, `numpy.exp()` sometimes overflows. If it happens, the agent is no longer able to make correct dicisions.
