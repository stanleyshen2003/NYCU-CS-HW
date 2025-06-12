import random
from typing import Dict, Any, Tuple, Optional, SupportsFloat, Union
import gymnasium
import numpy as np
from gymnasium.core import ObsType, ActType, RenderFrame
from racecar_gym.bullet.positioning import RecoverPositioningStrategy
from racecar_gym.envs.scenarios import SingleAgentScenario


# noinspection PyProtectedMember
class SingleAgentRaceEnv(gymnasium.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array_follow', 'rgb_array_birds_eye', 'rgb_array_lidar']
    }

    def __init__(self, scenario: str, render_mode: str = 'human', render_options: Optional[Dict[str, Any]] = None,
                 reset_when_collision: bool = True,
                 collision_penalty_weight: float = 10.):
        self.scenario_name = scenario
        scenario = SingleAgentScenario.from_spec(scenario, rendering=render_mode == 'human')
        self._scenario = scenario
        self._initialized = False
        self._render_mode = render_mode
        self._render_options = render_options or {}
        self.action_space = scenario.agent.action_space
        self.reset_when_collision = reset_when_collision

        self.recover_strategy = RecoverPositioningStrategy(progress_map=self._scenario.world._maps['progress'],
                                                           obstacle_map=self._scenario.world._maps['obstacle'],
                                                           alternate_direction=False)
        self.collision_penalties = []
        self.collision_penalty_weight = collision_penalty_weight

    @property
    def observation_space(self):
        space = self._scenario.agent.observation_space
        # space.spaces['time'] = gymnasium.spaces.Box(low=0, high=1, shape=(1,))
        return space

    @property
    def scenario(self):
        return self._scenario

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        assert self._initialized, 'Reset before calling step'
        observation, info = self._scenario.agent.step(action=action)

        self._scenario.world.update()

        cur_state = self._scenario.world.state()[self.scenario.agent._id]
        if self.reset_when_collision and self._scenario.agent.task._check_collision(cur_state):
            if 'austria' in self.scenario_name:
                cur_progress = cur_state['progress']
                collision_penalty = 30 + np.sum(cur_state['velocity'] ** 2) * self.collision_penalty_weight
                self.collision_penalties.append(collision_penalty)
                recover_pose = self.recover_strategy.get_recover_pose(cur_progress)
                self._scenario.agent._vehicle.reset(pose=recover_pose)
            else:
                raise ValueError('Recover are only supported for austria scenario')

        state = self._scenario.world.state()
        info = state[self._scenario.agent.id]
        if hasattr(self._scenario.agent.task, 'n_collision'):
            info['n_collision'] = self._scenario.agent.task.n_collision
        # observation['time'] = np.array([state[self._scenario.agent.id]['time']], dtype=np.float32)
        state[self._scenario.agent.id]['collision_penalties'] = np.array(self.collision_penalties)
        done = self._scenario.agent.done(state)
        reward = self._scenario.agent.reward(state, action)
        info['reward'] = reward
        return observation, reward, done, False, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[
        ObsType, Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.collision_penalties = []
        if not self._initialized:
            self._scenario.world.init()
            self._initialized = True
        else:
            self._scenario.world.reset()
        if options is not None and 'mode' in options:
            mode = options['mode']
        else:
            mode = 'grid'
        obs = self._scenario.agent.reset(self._scenario.world.get_starting_position(self._scenario.agent, mode))
        self._scenario.world.update()
        state = self._scenario.world.state()
        state[self._scenario.agent.id]['collision_penalties'] = np.array(self.collision_penalties)
        info = state[self._scenario.agent.id]
        if hasattr(self._scenario.agent.task, 'n_collision'):
            info['n_collision'] = self._scenario.agent.task.n_collision
        # obs['time'] = np.array(state[self._scenario.agent.id]['time'], dtype=np.float32)
        info['reward'] = 0.
        return obs, info

    def render(self) -> Union[RenderFrame, list[RenderFrame], None]:
        if self._render_mode == 'human':
            return None
        else:
            mode = self._render_mode.replace('rgb_array_', '')
            return self._scenario.world.render(mode=mode, agent_id=self._scenario.agent.id, **self._render_options)

    def force_render(self, render_mode: str, **kwargs):
        mode = render_mode.replace('rgb_array_', '')
        return self._scenario.world.render(mode=mode, agent_id=self._scenario.agent.id, **kwargs)
