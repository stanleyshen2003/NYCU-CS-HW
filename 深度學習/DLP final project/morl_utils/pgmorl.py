"""
PGMORL utility functions
reference: https://github.com/LucasAlegre/morl-baselines
"""
import time
from copy import deepcopy
from typing import List, Optional, Tuple, Union
import numpy.typing as npt
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from pymoo.indicators.hv import HV

def hypervolume(ref_point: np.ndarray, points: List[npt.ArrayLike]) -> float:
    """Computes the hypervolume metric for a set of points (value vectors) and a reference point (from Pymoo).

    Args:
        ref_point (np.ndarray): Reference point
        points (List[np.ndarray]): List of value vectors

    Returns:
        float: Hypervolume metric
    """
    return HV(ref_point=ref_point * -1)(np.array(points) * -1)

def sparsity(front: List[np.ndarray]) -> float:
    """Sparsity metric from PGMORL.

    Basically, the sparsity is the average distance between each point in the front.

    Args:
        front: current pareto front to compute the sparsity on

    Returns:
        float: sparsity metric
    """
    if len(front) < 2:
        return 0.0

    sparsity_value = 0.0
    m = len(front[0])
    front = np.array(front)
    for dim in range(m):
        objs_i = np.sort(deepcopy(front.T[dim]))
        for i in range(1, len(objs_i)):
            sparsity_value += np.square(objs_i[i] - objs_i[i - 1])
    sparsity_value /= len(front) - 1

    return sparsity_value

class PerformancePredictor:
    """Performance prediction model.

    Stores the performance deltas along with the used weights after each generation.
    Then, uses these stored samples to perform a regression for predicting the performance of using a given weight
    to train a given policy.
    Predicts: Weight & performance -> delta performance
    """

    def __init__(
        self,
        neighborhood_threshold: float = 0.1,
        sigma: float = 0.03,
        A_bound_min: float = 1.0,
        A_bound_max: float = 500.0,
        f_scale: float = 20.0,
    ):
        """Initialize the performance predictor.

        Args:
            neighborhood_threshold: The threshold for the neighborhood of an evaluation.
            sigma: The sigma value for the prediction model
            A_bound_min: The minimum value for the A parameter of the prediction model.
            A_bound_max: The maximum value for the A parameter of the prediction model.
            f_scale: The scale value for the prediction model.
        """
        # Memory
        self.previous_performance = []
        self.next_performance = []
        self.used_weight = []

        # Prediction model parameters
        self.neighborhood_threshold = neighborhood_threshold
        self.A_bound_min = A_bound_min
        self.A_bound_max = A_bound_max
        self.f_scale = f_scale
        self.sigma = sigma

    def add(self, weight: np.ndarray, eval_before_pg: np.ndarray, eval_after_pg: np.ndarray) -> None:
        """Add a new sample to the performance predictor.

        Args:
            weight: The weight used to train the policy.
            eval_before_pg: The evaluation before training the policy.
            eval_after_pg: The evaluation after training the policy.

        Returns:
            None
        """
        self.previous_performance.append(eval_before_pg)
        self.next_performance.append(eval_after_pg)
        self.used_weight.append(weight)

    def __build_model_and_predict(
        self,
        training_weights,
        training_deltas,
        training_next_perfs,
        current_dim,
        current_eval: np.ndarray,
        weight_candidate: np.ndarray,
        sigma: float,
    ):
        """Uses the hyperbolic model on the training data: weights, deltas and next_perfs to predict the next delta given the current evaluation and weight.

        Returns:
             The expected delta from current_eval by using weight_candidate.
        """

        def __f(x, A, a, b, c):
            return A * (np.exp(a * (x - b)) - 1) / (np.exp(a * (x - b)) + 1) + c

        def __hyperbolic_model(params, x, y):
            # f = A * (exp(a(x - b)) - 1) / (exp(a(x - b)) + 1) + c
            return (
                params[0] * (np.exp(params[1] * (x - params[2])) - 1.0) / (np.exp(params[1] * (x - params[2])) + 1)
                + params[3]
                - y
            ) * w

        def __jacobian(params, x, y):
            A, a, b, _ = params[0], params[1], params[2], params[3]
            J = np.zeros([len(params), len(x)])
            # df_dA = (exp(a(x - b)) - 1) / (exp(a(x - b)) + 1)
            J[0] = ((np.exp(a * (x - b)) - 1) / (np.exp(a * (x - b)) + 1)) * w
            # df_da = A(x - b)(2exp(a(x-b)))/(exp(a(x-b)) + 1)^2
            J[1] = (A * (x - b) * (2.0 * np.exp(a * (x - b))) / ((np.exp(a * (x - b)) + 1) ** 2)) * w
            # df_db = A(-a)(2exp(a(x-b)))/(exp(a(x-b)) + 1)^2
            J[2] = (A * (-a) * (2.0 * np.exp(a * (x - b))) / ((np.exp(a * (x - b)) + 1) ** 2)) * w
            # df_dc = 1
            J[3] = w

            return np.transpose(J)

        train_x = []
        train_y = []
        w = []
        for i in range(len(training_weights)):
            train_x.append(training_weights[i][current_dim])
            train_y.append(training_deltas[i][current_dim])
            diff = np.abs(training_next_perfs[i] - current_eval)
            dist = np.linalg.norm(diff / np.abs(current_eval))
            coef = np.exp(-((dist / sigma) ** 2) / 2.0)
            w.append(coef)

        train_x = np.array(train_x)
        train_y = np.array(train_y)
        w = np.array(w)

        A_upperbound = np.clip(np.max(train_y) - np.min(train_y), 1.0, 500.0)
        initial_guess = np.ones(4)
        res_robust = least_squares(
            __hyperbolic_model,
            initial_guess,
            loss="soft_l1",
            f_scale=self.f_scale,
            args=(train_x, train_y),
            jac=__jacobian,
            bounds=([0, 0.1, -5.0, -500.0], [A_upperbound, 20.0, 5.0, 500.0]),
        )

        return __f(weight_candidate[current_dim], *res_robust.x)

    def predict_next_evaluation(self, weight_candidate: np.ndarray, policy_eval: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the next evaluation of the policy.

        Use a part of the collected data (determined by the neighborhood threshold) to predict the performance
        after using weight to train the policy whose current evaluation is policy_eval.

        Args:
            weight_candidate: weight candidate
            policy_eval: current evaluation of the policy

        Returns:
            the delta prediction, along with the predicted next evaluations
        """
        neighbor_weights = []
        neighbor_deltas = []
        neighbor_next_perf = []
        current_sigma = self.sigma / 2.0
        current_neighb_threshold = self.neighborhood_threshold / 2.0
        # Iterates until we find at least 4 neighbors, enlarges the neighborhood at each iteration
        while len(neighbor_weights) < 4:
            # Enlarging neighborhood
            current_sigma *= 2.0
            current_neighb_threshold *= 2.0

            # print(f"current_neighb_threshold: {current_neighb_threshold}")
            # print(f"np.abs(policy_eval): {np.abs(policy_eval)}")
            if current_neighb_threshold == np.inf or current_sigma == np.inf:
                raise ValueError("Cannot find at least 4 neighbors by enlarging the neighborhood.")

            # Filtering for neighbors
            for previous_perf, next_perf, neighb_w in zip(self.previous_performance, self.next_performance, self.used_weight):
                if np.all(np.abs(previous_perf - policy_eval) < current_neighb_threshold * np.abs(policy_eval)) and tuple(
                    next_perf
                ) not in list(map(tuple, neighbor_next_perf)):
                    neighbor_weights.append(neighb_w)
                    neighbor_deltas.append(next_perf - previous_perf)
                    neighbor_next_perf.append(next_perf)

        # constructing a prediction model for each objective dimension, and using it to construct the delta predictions
        delta_predictions = [
            self.__build_model_and_predict(
                training_weights=neighbor_weights,
                training_deltas=neighbor_deltas,
                training_next_perfs=neighbor_next_perf,
                current_dim=obj_num,
                current_eval=policy_eval,
                weight_candidate=weight_candidate,
                sigma=current_sigma,
            )
            for obj_num in range(weight_candidate.size)
        ]
        delta_predictions = np.array(delta_predictions)
        return delta_predictions, delta_predictions + policy_eval

def generate_weights(delta_weight: float) -> np.ndarray:
    """Generates weights uniformly distributed over the objective dimensions. These weight vectors are separated by delta_weight distance.

    Args:
        delta_weight: distance between weight vectors
    Returns:
        all the candidate weights
    """
    return np.linspace((0.0, 1.0), (1.0, 0.0), int(1 / delta_weight) + 1, dtype=np.float32)

class PerformanceBuffer:
    """Stores the population. Divides the objective space in to n bins of size max_size.

    (!) restricted to 2D objective space (!)
    """

    def __init__(self, num_bins: int, max_size: int, origin: np.ndarray):
        """Initializes the buffer.

        Args:
            num_bins: number of bins
            max_size: maximum size of each bin
            origin: origin of the objective space (to have only positive values)
        """
        self.num_bins = num_bins
        self.max_size = max_size
        self.origin = -origin
        self.dtheta = np.pi / 2.0 / self.num_bins
        self.bins = [[] for _ in range(self.num_bins)]
        self.bins_evals = [[] for _ in range(self.num_bins)]

    @property
    def evaluations(self) -> List[np.ndarray]:
        """Returns the evaluations of the individuals in the buffer."""
        # flatten
        return [e for l in self.bins_evals for e in l]

    @property
    def individuals(self) -> list:
        """Returns the individuals in the buffer."""
        return [i for l in self.bins for i in l]

    def add(self, candidate, evaluation: np.ndarray):
        """Adds a candidate to the buffer.

        Args:
            candidate: candidate to add
            evaluation: evaluation of the candidate
        """

        def center_eval(eval):
            # Objectives must be positive
            return np.clip(eval + self.origin, 0.0, float("inf"))

        centered_eval = center_eval(evaluation)
        norm_eval = np.linalg.norm(centered_eval)
        theta = np.arccos(np.clip(centered_eval[1] / (norm_eval + 1e-3), -1.0, 1.0))
        buffer_id = int(theta // self.dtheta)

        if buffer_id < 0 or buffer_id >= self.num_bins:
            return

        if len(self.bins[buffer_id]) < self.max_size:
            self.bins[buffer_id].append(deepcopy(candidate))
            self.bins_evals[buffer_id].append(evaluation)
        else:
            for i in range(len(self.bins[buffer_id])):
                stored_eval_centered = center_eval(self.bins_evals[buffer_id][i])
                if np.linalg.norm(stored_eval_centered) < np.linalg.norm(centered_eval):
                    self.bins[buffer_id][i] = deepcopy(candidate)
                    self.bins_evals[buffer_id][i] = evaluation
                    break

def get_non_pareto_dominated_inds(candidates: Union[np.ndarray, List], remove_duplicates: bool = True) -> np.ndarray:
    """A batched and fast version of the Pareto coverage set algorithm.

    Args:
        candidates (ndarray): A numpy array of vectors.
        remove_duplicates (bool, optional): Whether to remove duplicate vectors. Defaults to True.

    Returns:
        ndarray: The indices of the elements that should be kept to form the Pareto front or coverage set.
    """
    candidates = np.array(candidates)
    uniques, indcs, invs, counts = np.unique(candidates, return_index=True, return_inverse=True, return_counts=True, axis=0)

    res_eq = np.all(candidates[:, None, None] <= candidates, axis=-1).squeeze()
    res_g = np.all(candidates[:, None, None] < candidates, axis=-1).squeeze()
    c1 = np.sum(res_eq, axis=-1) == counts[invs]
    c2 = np.any(~res_g, axis=-1)
    if remove_duplicates:
        to_keep = np.zeros(len(candidates), dtype=bool)
        to_keep[indcs] = 1
    else:
        to_keep = np.ones(len(candidates), dtype=bool)

    return np.logical_and(c1, c2) & to_keep


def filter_pareto_dominated(candidates: Union[np.ndarray, List], remove_duplicates: bool = True) -> np.ndarray:
    """A batched and fast version of the Pareto coverage set algorithm.

    Args:
        candidates (ndarray): A numpy array of vectors.
        remove_duplicates (bool, optional): Whether to remove duplicate vectors. Defaults to True.

    Returns:
        ndarray: A Pareto coverage set.
    """
    candidates = np.array(candidates)
    if len(candidates) < 2:
        return candidates
    return candidates[get_non_pareto_dominated_inds(candidates, remove_duplicates=remove_duplicates)]

def filter_convex_dominated(candidates: Union[np.ndarray, List]) -> np.ndarray:
    """A fast version to prune a set of points to its convex hull. This leverages the QuickHull algorithm.

    This algorithm first computes the convex hull of the set of points and then prunes the Pareto dominated points.

    Args:
        candidates (ndarray): A numpy array of vectors.

    Returns:
        ndarray: A convex coverage set.
    """
    candidates = np.array(candidates)
    if len(candidates) > 2:
        hull = ConvexHull(candidates)
        ccs = candidates[hull.vertices]
    else:
        ccs = candidates
    return filter_pareto_dominated(ccs)


class ParetoArchive:
    """Pareto archive."""

    def __init__(self, convex_hull: bool = False):
        """Initializes the Pareto archive."""
        self.convex_hull = convex_hull
        self.individuals: list = []
        self.evaluations: List[np.ndarray] = []

    def add(self, candidate, evaluation: np.ndarray):
        """Adds the candidate to the memory and removes Pareto inefficient points.

        Args:
            candidate: The candidate to add.
            evaluation: The evaluation of the candidate.
        """
        self.evaluations.append(evaluation)
        self.individuals.append(deepcopy(candidate))

        # Non-dominated sorting
        if self.convex_hull:
            nd_candidates = {tuple(x) for x in filter_convex_dominated(self.evaluations)}
        else:
            nd_candidates = {tuple(x) for x in filter_pareto_dominated(self.evaluations)}

        # Reconstruct the pareto archive (because Non-Dominated sorting might change the order of candidates)
        non_dominated_evals = []
        non_dominated_evals_tuples = []
        non_dominated_individuals = []
        for e, i in zip(self.evaluations, self.individuals):
            if tuple(e) in nd_candidates and tuple(e) not in non_dominated_evals_tuples:
                non_dominated_evals.append(e)
                non_dominated_evals_tuples.append(tuple(e))
                non_dominated_individuals.append(i)
        self.evaluations = non_dominated_evals
        self.individuals = non_dominated_individuals