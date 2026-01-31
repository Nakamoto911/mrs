"""
Models Module
=============
Linear, tree-based, and neural network models.
"""

from .linear_models import (
    LinearModelWrapper,
    VECMWrapper,
    create_linear_model
)

from .tree_models import (
    TreeModelWrapper,
    RegimeClassifier,
    create_tree_model
)

from .neural_nets import (
    MLPWrapper,
    LSTMWrapper,
    create_neural_model
)

from .lstm_validation import (
    LSTMSequenceValidator,
    SequenceRequirements,
    SequenceValidationResult
)

__all__ = [
    'LinearModelWrapper',
    'VECMWrapper',
    'create_linear_model',
    'TreeModelWrapper',
    'RegimeClassifier',
    'create_tree_model',
    'MLPWrapper',
    'LSTMWrapper',
    'create_neural_model',
    'LSTMSequenceValidator',
    'SequenceRequirements',
    'SequenceValidationResult'
]
