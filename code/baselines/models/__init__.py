"""Model registry of the paper torque baselines.

mamba2 is imported lazily so the other models run without mamba_ssm.
"""
from .mlp import MLP
from .fconv import FConv
from .lstm import BiLSTMModel
from .gru import GRUFallback
from .transformer import TransformerModel

MODEL_REGISTRY = {
    "mlp": MLP,
    "fconv": FConv,
    "lstm": BiLSTMModel,
    "gru": GRUFallback,
    "transformer": TransformerModel,
}


def build_model(name: str, input_dim: int, output_dim: int):
    if name == "mamba2":
        from .mamba2 import Mamba2Model
        return Mamba2Model(input_dim, output_dim)
    return MODEL_REGISTRY[name](input_dim, output_dim)
