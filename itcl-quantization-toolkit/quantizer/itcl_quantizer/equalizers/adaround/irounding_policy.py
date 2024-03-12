from abc import ABCMeta, abstractmethod

import numpy as np


class IRoundingPolicy(metaclass=ABCMeta):
    @property
    @abstractmethod
    def rounding_policy(
        self,
    ) -> np.ndarray:
        """Set a binary numpy array with the scale and zp

        Returns:
            np.ndarray: _description_
        """

    @rounding_policy.setter
    @abstractmethod
    def rounding_policy(self, policy: np.ndarray):
        ...

    @staticmethod
    def _get_base_rounding(tensor: np.ndarray, scale: float) -> np.ndarray:
        """Get the Round To Nearest rounding policy of a float tensor

        Args:
            tensor (np.ndarray): Tensor to get the policy from.

        Returns:
            np.ndarray: _description_
        """
        tensor = tensor / scale
        return np.where(np.floor(tensor + 0.5) > tensor, 1, 0)
