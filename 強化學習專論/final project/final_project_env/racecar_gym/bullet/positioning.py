import random
from abc import ABC

from racecar_gym.core.definitions import Pose
from racecar_gym.core.gridmaps import GridMap
import numpy as np


class PositioningStrategy(ABC):

    def get_pose(self, agent_index: int) -> Pose:
        pass


class AutomaticGridStrategy(PositioningStrategy):
    def __init__(self, obstacle_map: GridMap, number_of_agents: int):
        self._distance_map = obstacle_map
        self._number_of_agents = number_of_agents

    def get_pose(self, agent_index: int) -> Pose:
        px, py = self._distance_map.to_pixel(position=(0, 0, 0))
        starting_area = self._distance_map.map[px:px + 20, py - 20:py + 20]
        center = np.argmax(starting_area)
        max_index = np.unravel_index(center, shape=starting_area.shape)
        center_position = self._distance_map.to_meter(px + max_index[0], py + max_index[1])
        if agent_index % 2 == 0:
            y = center_position[1] + 0.4
        else:
            y = center_position[1] - 0.4

        x = center_position[0] + 1.0 * (self._number_of_agents - agent_index) / 2

        return (x, y, 0.05), (0.0, 0.0, 0.0)


class RandomPositioningStrategy(PositioningStrategy):

    def __init__(self, progress_map: GridMap, obstacle_map: GridMap,
                 min_distance_to_obstacle: float = 0.5, alternate_direction=False):
        self._progress = progress_map
        self._obstacles = obstacle_map
        self._obstacle_margin = min_distance_to_obstacle
        self._alternate_direction = alternate_direction

    def get_pose(self, agent_index: int) -> Pose:
        center_corridor = np.argwhere(self._obstacles.map > self._obstacle_margin)
        delta_progress_next_pos = 0.025
        if self._alternate_direction and random.random() < 0.5:
            delta_progress_next_pos = -delta_progress_next_pos
        x, y, angle = self._random_position(self._progress, center_corridor, delta_progress_next_pos)
        return (x, y, 0.05), (0, 0, angle)

    def _random_position(self, progress_map, sampling_map, delta_progress_next_pos=0.025):
        position = random.choice(sampling_map)
        progress = progress_map.map[position[0], position[1]]

        # next position is a random point in the map which has progress greater than the current `position`
        # note: this approach suffers in "wide" tracks
        if delta_progress_next_pos > 0:
            direction_progress_min = (progress + delta_progress_next_pos) % 1
            direction_progress_max = (progress + 2 * delta_progress_next_pos) % 1
        else:
            direction_progress_min = (progress + 2 * delta_progress_next_pos) % 1
            direction_progress_max = (progress + delta_progress_next_pos) % 1
        # consider corner cases between end of track end begin
        if direction_progress_min < direction_progress_max:
            direction_area = np.argwhere(np.logical_and(
                progress_map.map > direction_progress_min,
                progress_map.map <= direction_progress_max,
            ))
        else:
            # bugfix: when min=0.99, max=0.01, the 'and' is empty
            direction_area = np.argwhere(np.logical_or(
                progress_map.map <= direction_progress_min,
                progress_map.map > direction_progress_max,
            ))
        if direction_area.shape[0] == 0:
            raise ValueError(
                f"starting position not exist, consider to change `delta_progress`={delta_progress_next_pos}")
        next_position = random.choice(direction_area)
        px, py = position[0], position[1]
        npx, npy = next_position[0], next_position[1]
        diff = np.array(progress_map.to_meter(npx, npy)) - np.array(progress_map.to_meter(px, py))
        angle = np.arctan2(diff[1], diff[0])
        x, y = progress_map.to_meter(px, py)
        return x, y, angle


class RandomPositioningWithinBallStrategy(RandomPositioningStrategy):

    def __init__(self, progress_map: GridMap, obstacle_map: GridMap, drivable_map: np.ndarray,
                 min_distance_to_obstacle: float = 0.5, progress_center: float = 0.0, progress_radius: float = 0.01):
        self._progress = progress_map
        self._obstacles = obstacle_map
        self._occupancy = drivable_map
        self._obstacle_margin = min_distance_to_obstacle
        self._progress_center = progress_center  # the interval is centered w.r.t. a fixed progress value
        self._progress_radius = progress_radius  # the width of the interval is +-progress_radius

    def get_pose(self, agent_index: int) -> Pose:
        progress_interval = np.logical_and(self._progress.map >= self._progress_center - self._progress_radius,
                                           self._progress.map <= self._progress_center + self._progress_radius)
        center_corridor = np.argwhere(np.logical_and(progress_interval,
                                                     (self._occupancy * self._obstacles.map) >= self._obstacle_margin))
        x, y, angle = self._random_position(self._progress, center_corridor)
        return (x, y, 0.05), (0, 0, angle)


class RecoverPositioningStrategy:

    def __init__(self, progress_map: GridMap, obstacle_map: GridMap,
                 min_distance_to_obstacle: float = 0.69, alternate_direction=False):
        self._progress = progress_map
        self._obstacles = obstacle_map
        self._obstacle_margin = min_distance_to_obstacle
        self._alternate_direction = alternate_direction

    def get_recover_pose(self, cur_progress):
        # 1. Get current progress
        # 2. Get the corresponding checkpoint


        # Offset: [950:1450, 800:1400]
        checkpoints = [
            (56, 279, 20, 0),  # 0
            (56, 359, 20, 0),  # 1
            (126, 427, 5, -20),  # 2
            (206, 450, 4, -20),  # 3
            (285, 467, 5, -20),  # 4
            (365, 496, 8, -20),  # 5
            (444, 493, -10, 3),  # 6
            (418, 413, -10, 4),  # 7
            (383, 334, -10, 3),  # 8
            (355, 255, -10, 3),  # 9
            (287, 265, 10, 0),  # 10
            (310, 344, 10, -3),  # 11
            (272, 410, -2, 10),  # 12
            (193, 360, -10, -4),  # 13
            (223, 280, -10, 3),  # 14
            (204, 201, -10, 3),  # 15
            (184, 121, -10, 3),  # 16
            (118, 58, 0, 10),  # 17
            (57, 120, 10, 0),  # 18
            (56, 199, 10, 0),  # 19
        ]  # x (right),y (up)
        checkpoint_dict = {i: c for i, c in  enumerate(checkpoints)}
        cur_idx = int(np.ceil(cur_progress * 20.)) if cur_progress > 0. else 1
        checkpoint = checkpoint_dict[cur_idx - 1]
        progress_map = self._progress
        position = (950+checkpoint[0], 800+checkpoint[1], checkpoint[2], checkpoint[3])
        px, py = position[0], position[1]
        npx, npy = position[0]-position[3],position[1]+position[2]
        diff = np.array(progress_map.to_meter(npx, npy)) - np.array(progress_map.to_meter(px, py))
        angle = np.arctan2(diff[1], diff[0])
        x, y = progress_map.to_meter(px, py)
        # print(f'px,py | npx,npy | angle ---> {progress_map.to_meter(px, py)} | {progress_map.to_meter(npx, npy)}  | {angle}')
        return (x, y, 0.05), (0, 0, angle)

    def get_recover_pose2(self, cur_pose, delta_progress_next_pos=0.025) -> Pose:
        progress_map = self._progress


        # Available corridor
        center_corridor = np.argwhere(self._obstacles.map > self._obstacle_margin)
        # Nearest
        x, y, *_ = cur_pose
        # x, y = self._progress.to_meter(x, y)
        corridor_xy = np.array(list(map(lambda pair: progress_map.to_meter(*pair),center_corridor)))
        dist_square = ((corridor_xy[:, 0] - x) ** 2) + (corridor_xy[:, 1] - y) ** 2
        min_indices = np.where(dist_square == dist_square.min())
        min_idx = random.choice(min_indices)[0]
        position = center_corridor[min_idx]
        # print(f'_____ (x,y) = ({x:.3f},{y:.3f}); recover ({position[0]:.3f},{position[1]:.3f})')

        #

        progress = progress_map.map[position[0], position[1]]

        # next position is a random point in the map which has progress greater than the current `position`
        # note: this approach suffers in "wide" tracks
        if delta_progress_next_pos > 0:
            direction_progress_min = (progress + delta_progress_next_pos) % 1
            direction_progress_max = (progress + 2 * delta_progress_next_pos) % 1
        else:
            direction_progress_min = (progress + 2 * delta_progress_next_pos) % 1
            direction_progress_max = (progress + delta_progress_next_pos) % 1
        # consider corner cases between end of track end begin
        if direction_progress_min < direction_progress_max:
            direction_area = np.argwhere(np.logical_and(
                progress_map.map > direction_progress_min,
                progress_map.map <= direction_progress_max,
            ))
        else:
            # bugfix: when min=0.99, max=0.01, the 'and' is empty
            direction_area = np.argwhere(np.logical_or(
                progress_map.map <= direction_progress_min,
                progress_map.map > direction_progress_max,
            ))
        if direction_area.shape[0] == 0:
            raise ValueError(
                f"starting position not exist, consider to change `delta_progress`={delta_progress_next_pos}")

        # Select the farthest point
        area_xy = np.array(list(map(lambda pair: progress_map.to_meter(*pair), direction_area)))
        dist_square = ((area_xy[:, 0] - x) ** 2) + (area_xy[:, 1] - y) ** 2
        min_indices = np.where(dist_square == dist_square.min())
        min_idx = random.choice(min_indices)[0]
        next_position = direction_area[min_idx]

        # next_position = random.choice(direction_area)
        px, py = position[0], position[1]
        npx, npy = next_position[0], next_position[1]
        diff = np.array(progress_map.to_meter(npx, npy)) - np.array(progress_map.to_meter(px, py))
        angle = np.arctan2(diff[1], diff[0])
        x, y = progress_map.to_meter(px, py)
        print(f'px,py | npx,npy | angle ---> {progress_map.to_meter(px, py)} | {progress_map.to_meter(npx, npy)}  | {angle}')

        return (x, y, 0.05), (0, 0, angle)
