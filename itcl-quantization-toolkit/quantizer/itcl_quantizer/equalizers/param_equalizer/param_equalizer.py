from typing import Callable
from itcl_quantizer.equalizers.param_equalizer.abstract_param_optimizer import (
    AbstractParamOptimizer,
)
from itcl_quantizer.util.network import Network
from itcl_inference_engine.network.Network import Network as NetworkIE


class ParamEqualizerNet:
    """Parameter Equalizer Orchestrator
    Equalizes the Scales and Zero Points of an entire network to improve the network overall loss.
    This class updates the scales and zero points of the network nodes by reference.
    """

    def __init__(
        self,
        net: Network,
        loss_fn: Callable[[NetworkIE], float],
        optimizer_factory: Callable[[], AbstractParamOptimizer],
    ):
        """
        Param Equalizer Constructor


        Args:
            net (Network): Full Network (T)
            loss_fn (Callable[[NetworkIE], float]): _description_
            optimizer_factory (Callable[[], AbstractParamOptimizer]): _description_
        """
        self._loss_fn = loss_fn
        self._net = net
        self._layer_results = net.as_quant_results()
        self._optimizer_factory = optimizer_factory

    def equalize(self):
        """
        Main Function. Equalizes each layer individually.
        """
        i = 0
        results = list(self._layer_results)
        results.reverse()
        for result in results:
            i = i + 1
            if i in [1, 2]:
                continue
            result.layer.param_equalizer(
                self._optimizer_factory,
                result,
                lambda: self._loss_fn(self._net.as_sequential_network()),
            )
