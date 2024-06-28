from abc import ABC, abstractmethod


class Environment(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_rgbd_image(self):
        pass

    @abstractmethod
    def project_to_3d(self, points, depth):
        pass

    @abstractmethod
    def move_in_camera_frame(self, t, R):
        pass

    @abstractmethod
    def record_demo(self):
        pass

    @abstractmethod
    def replay_demo(self, demo_velocities):
        pass
