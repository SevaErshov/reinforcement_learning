import numpy as np


def get_trajectory(env, agent, max_len=1000):
    """
    Args:
        env (gym.Env)
        agent: object with method 'get_action' implemented
        max_len: maximum trajectory length
    Returns:
        Dict with lists of states, actions and rewards
    """
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    state, info = env.reset()

    for _ in range(max_len):
        trajectory['states'].append(state)
        
        action = agent.get_action(state, info)
        trajectory['actions'].append(action)
        
        state, reward, terminated, truncated, info = env.step(action)
        trajectory['rewards'].append(reward)
        
        if terminated or truncated:
            break
    
    return trajectory


def evaluate_agent(env, agent, iteration_n=10):
    """
    Args:
        env (gym.Env)
        agent: object with method 'get_action' implemented
        iteration_n: number of iterations to evaluate the agent
    Returns:
        Mean and standard deviation of total rewards
    """
    total_rewards = []
    for _ in range(iteration_n):
        rewards = get_trajectory(env, agent)['rewards']
        total_rewards.append(np.sum(rewards))
        
    return np.mean(total_rewards), np.std(total_rewards)
