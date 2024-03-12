from __future__ import annotations
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

from typing import Callable, List, Optional

import numpy as np


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid Circular Imports
    from itcl_quantizer.equalizers.adaround.iround_optimizer import IRoundOptimizer
    from itcl_quantizer.equalizers.param_equalizer.abstract_param_optimizer import (
        AbstractParamOptimizer,
    )

    from itcl_quantizer.tensor_extractor.operator import Operator
    from itcl_quantizer.tensor_extractor.tensor import NodeTensorBase


@dataclass
class QuantizationResult:
    """Quantization Result: Stores the results of quantized layer"""

    input_data: np.ndarray
    """Float Input Data: A representative sample of the expected layer output. This data is the representative dataset
    and it changes after each layer. 
    """

    operators: Optional[List[Operator[NodeTensorBase, NodeTensorBase]]]
    """Generated Operators: A Dense Layer generates FullyConnected and ActivationFn Operators.
    """

    out_node: Optional[NodeTensorBase]
    """The last node of the last operator. It will be used as the Input Node of the next layer.
    """

    layer: "AbstractLayer"
    """Current Layer Operator that has generated this QuantizationResult
    """


class AbstractLayer(metaclass=ABCMeta):
    """Abstract Layer that implement the default behavior

    Args:
        metaclass (_type_, optional): _description_. Defaults to ABCMeta.

    Raises:
        NotImplemented: _description_
    """

    @abstractmethod
    def quantize(self, input_result: QuantizationResult) -> QuantizationResult:
        """Method that generates the quantization result of the layer
        Args:
            input_result(QuantizationResult): previous Network Quantization Result
        Returns:
            QuantizationResult: Quantization Result of the layer
        """

    def adaround(
        self,
        optimizer_factory: Callable[[], IRoundOptimizer],
        results: QuantizationResult,
        cost_fn: Callable[[], float],
    ):
        """Method that applies the AdaRound to the current Quantization Results associated with the specified layer.

        By default, calling this method does nothing unless it is overridden by the layer implementation


        Args:
            optimizer (IRoundOptimizer): A Rounding Optimizer, for example RoundingAnnealer
            results (QuantizationResult): Quantization Results to be adarounded
            cost_fn (Callable[[], float]): The cost function
        """

    def param_equalizer(
        self,
        optimizer_factory: Callable[[], AbstractParamOptimizer],
        results: QuantizationResult,
        cost_fn: Callable[[], float],
    ):
        """Quantization Parameter (scale & zp) equalizer
        Tweaks the scale and zp of multiple results tensors to reduce the cost_fn

        By default, calling this method does nothing unless it is overridden by the layer implementation

        Args:
            optimizer (AbstractParamOptimizer): The optimizer to be used
            results (QuantizationResult): The layer quantization results to be optimized
            cost_fn (Callable[[], float]): The cost Function
        """
