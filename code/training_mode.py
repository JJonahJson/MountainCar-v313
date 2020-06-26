from enum import Enum, auto

class TrainingMode(Enum):
    DQN = auto()
    FixedQTargets = auto()
    DoubleDQN = auto()
