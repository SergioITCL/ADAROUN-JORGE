from typing import Callable, List, Tuple
import numpy as np
from abc import ABCMeta, abstractmethod


class IRoundOptimizer(metaclass=ABCMeta):

    """Interface that declares all the methods a Rounding Optimizer should include."""

    @abstractmethod
    def set_cost_fn(self, fn: Callable[[List[np.ndarray]], float]) -> "IRoundOptimizer":
        """Updates the cost function, this function receives the updated rounding policies.

        Args:
            fn (Callable[[List[np.ndarray]], float]): The new cost function


        Returns:
            IRoundOptimizer: Self class
        """

    @abstractmethod
    def set_initial_neigh(self, neigh: List[np.ndarray]) -> "IRoundOptimizer":
        """Initializes the rounding policy neighborhood to optimize.

        Args:
            neigh (List[np.ndarray]): A list of binary numpy ndarrays.
            The arrays can have different shapes.

        Returns:
            IRoundOptimizer: Self class
        """

    @abstractmethod
    def optimize(self) -> Tuple[List[np.ndarray], float]:
        """Optimization Method

        Returns:
            Tuple[List[np.ndarray], float]: Returns the optimized rounding policy and
             the final loss/cost
        """
