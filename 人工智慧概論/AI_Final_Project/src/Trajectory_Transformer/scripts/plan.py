import json
import pdb
import os
import sys
from os.path import join
import gym
import gym_anytrading
import numpy as np
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions 
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import trajectory.utils as utils
import trajectory.datasets as datasets
from trajectory.datasets.Random.buildEnv import createEnv
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from trajectory.search import (
    beam_plan,
    make_prefix,
    extract_actions,
    update_context,
)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'


class Parser(utils.Parser):
    dataset: str = 'DDQN_10_2330'
    config: str = 'config.offline'

#######################
######## setup ########
#######################

args = Parser().parse_args('plan')
code = '2330'
img_save_path = 'Img'


#######################
####### models ########
#######################


dataset = utils.load_from_config(args.logbase, args.dataset, args.gpt_loadpath,'data_config.pkl')



#######################
####### dataset #######
#######################
def state_preprocess(state):
    tempstate = state
    for i in range(12):
        for j in range(4):
            tempstate[i*4+j] = (state[44+j] - state[i*4+j])/state[44+j]
    return tempstate


#renderer = utils.make_renderer(args)
timer = utils.timer.Timer()

discretizer = dataset.discretizer
discount = dataset.discount
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

value_fn = lambda x: discretizer.value_fn(x, args.percentile)
#preprocess_fn = datasets.get_preprocess_fn(env.name)

#######################
###### main loop ######
#######################
w = SummaryWriter('tb_record_1/comp_profit/TT')
for test_ep in [500]:
    env = createEnv(code, frame_bounds=(1000, 1500))
    env.seed(test_ep)
    observation = env.reset()
    observation = observation.reshape(-1)
    total_reward = 0
    observation = state_preprocess(observation)
    rollout = [observation.copy()]

    context = []
    tb_save_path = 'tb_record_1/TT_' + args.dataset + '_{}'.format(test_ep)
    gpt, gpt_epoch = utils.load_model(
        args.logbase, 
        args.dataset, 
        args.gpt_loadpath,
        epoch=test_ep, 
        device=args.device)
    reward_writer = SummaryWriter(tb_save_path)
    T = 1187
    for t in range(T):

        #observation = preprocess_fn(observation)
        #observation = observation.reshape(-1)
        #observation = state_preprocess(observation)

        if t % args.plan_freq == 0:
            ## concatenate previous transitions and current observations to input to model
            prefix = make_prefix(discretizer, context, observation, args.prefix_context)

            ## sample sequence from model beginning with `prefix`
            sequence = beam_plan(
                gpt, value_fn, prefix,
                args.horizon, args.beam_width, args.n_expand, observation_dim, action_dim,
                discount, args.max_context_transitions, verbose=args.verbose,
                k_obs=args.k_obs, k_act=args.k_act, cdf_obs=args.cdf_obs, cdf_act=args.cdf_act,
            )

        else:
            sequence = sequence[1:]

        ## [ horizon x transition_dim ] convert sampled tokens to continuous trajectory
        sequence_recon = discretizer.reconstruct(sequence)

        ## [ action_dim ] index into sampled trajectory to grab first action
        action = extract_actions(sequence_recon, observation_dim, action_dim, t=0)
        
        ## execute action in environment
        next_observation, reward, terminal, info = env.step(np.argmax(action))
        w.add_scalar('Profit', env._total_profit, t)
        ## update return
        total_reward += reward
        #score = env.get_normalized_score(total_reward)

        ## update rollout observations and context transitions
        next_observation = state_preprocess(next_observation.reshape(-1))
        rollout.append(next_observation.copy())
        context = update_context(context, discretizer, observation, action, reward, args.max_context_transitions)
        print(
            f'[ plan ] t: {t} / {T} | r: {reward:.2f} | R: {total_reward:.2f} '
            f'time: {timer():.2f} | {args.dataset} | {args.exp_name} | {args.suffix}\n'
        )
        print(info, action)
        reward_writer.add_scalar('Total reward', total_reward, t)

        if terminal: 
            print(info)
            break

        observation = next_observation

    ## Show result
    #env.render_all()
    #env.save_rendering('Images/TT_' + args.dataset + '_{}.png'.format(test_ep))
    ## save result as a json file
    json_path = join(args.savepath, 'rollout.json')
    json_data = {'step': t, 'return': total_reward, 'term': terminal, 'gpt_epoch': gpt_epoch}
    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
