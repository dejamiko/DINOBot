from enum import Enum, auto


class AutoName(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()


class Task(AutoName):
    GRASPING = auto()
    PUSHING = auto()
    HAMMERING = auto()
