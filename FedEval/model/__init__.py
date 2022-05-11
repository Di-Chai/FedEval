# from .BaseModel import TFModel, KerasModel, parse_strategy_and_compress, recover_to_weights
# from .MLP import MLP, MLPAttack
# from .LeNet import LeNet, LeNetAttack
# from .MobileNet import MobileNet
# from .ResNet50 import ResNet50

import tensorflow
# Change the dtype
tensorflow.keras.backend.set_floatx('float64')

from .MLP import MLP
from .LeNet import LeNet
from .StackedLSTM import StackedLSTM
