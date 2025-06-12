from ppo_agent_atari_01 import AtariPPOAgent
import argparse
import json
import numpy as np
import requests


def connect(agent, url: str = 'http://localhost:5000'):
    while True:
        # Get the observation
        response = requests.get(f'{url}')
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break
        obs = json.loads(response.text)['observation']
        obs = np.array(obs).astype(np.uint8)

        # Decide an action based on the observation (Replace this with your RL agent logic)
        print(1)
        action_to_take = agent.act(obs)  # Replace with actual action
        if action_to_take[0] > 0:
            action_to_take[0] += 0.02
        # Send an action and receive new observation, reward, and done status
        response = requests.post(f'{url}', json={'action': action_to_take.tolist()})
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break

        result = json.loads(response.text)
        terminal = result['terminal']

        if terminal:
            print('Episode finished.')
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default='http://localhost:5000', help='The url of the server.')
    args = parser.parse_args()
    config = {
        "gpu": True,
        "training_steps": 1e8,
        "update_sample_count": 10000,
        "discount_factor_gamma": 0.99,
        "discount_factor_lambda": 0.95,
        "clip_epsilon": 0.2,
        "max_gradient_norm": 0.5,
        "batch_size": 128,
        "logdir": 'log/',
        "update_ppo_epoch": 3,
        "learning_rate": 2.5e-4,
        "value_coefficient": 0.5,
        "entropy_coefficient": 0.01,
        "horizon": 128,
        "env_id": 'ALE/Enduro-v5',  #Enduro-v5
        "eval_interval": 100,
        "eval_episode": 3,
    }
    agent = AtariPPOAgent(config)
    agent.load('log/new_actions/model_1553945_328.pth')

    connect(agent, url=args.url)
