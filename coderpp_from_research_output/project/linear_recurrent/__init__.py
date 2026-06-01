"""Linear Recurrent and Gated Architectures.

Provides four families of attention alternatives as drop-in building
blocks for sequence models:

- **RWKV**: WKV operator (exponential-decay weighted key-value),
  TimeMixBlock, ChannelMixBlock, RWKVBlock.
- **xLSTM**: mLSTMCell (matrix memory with covariance storage),
  sLSTMCell (scalar exponential-gated LSTM), xLSTMBlock.
- **Griffin**: SimpleRGLRU (real-gated linear recurrent unit),
  GriffinBlock.
- **RetNet**: retention_parallel, retention_recurrent,
  retention_chunkwise, MultiScaleRetention, RetNetBlock.

Shared utilities (RMSNorm, LayerNorm, activations, SwiGLU) live in
``common``.
"""

from .common import (
    RMSNorm,
    LayerNorm,
    SquaredReLU,
    SwiGLU,
    get_activation,
)

from .rwkv import (
    WKVOperator,
    TimeMixBlock,
    ChannelMixBlock,
    RWKVBlock,
    token_shift,
)

from .xlstm import (
    mLSTMCell,
    sLSTMCell,
    xLSTMBlock,
)

from .griffin import (
    RGLRU,
    SimpleRGLRU,
    GriffinBlock,
)

from .retnet import (
    retention_parallel,
    retention_recurrent,
    retention_chunkwise,
    MultiScaleRetention,
    RetNetBlock,
    _build_decay_matrix,
)

__all__ = [
    # Common
    "RMSNorm",
    "LayerNorm",
    "SquaredReLU",
    "SwiGLU",
    "get_activation",
    # RWKV
    "WKVOperator",
    "TimeMixBlock",
    "ChannelMixBlock",
    "RWKVBlock",
    "token_shift",
    # xLSTM
    "mLSTMCell",
    "sLSTMCell",
    "xLSTMBlock",
    # Griffin
    "RGLRU",
    "SimpleRGLRU",
    "GriffinBlock",
    # RetNet
    "retention_parallel",
    "retention_recurrent",
    "retention_chunkwise",
    "MultiScaleRetention",
    "RetNetBlock",
    "_build_decay_matrix",
]
