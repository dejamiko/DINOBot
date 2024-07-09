import numpy as np


class Config:
    SEED = 0
    # DINOBot deployment constants
    ERR_THRESHOLD = 0.02
    IMAGE_DIR = "images/"
    # using the ssh tunnel from home ssh -L 8080:linnet.doc.ic.ac.uk:8000 md1823@shell5.doc.ic.ac.uk
    # BASE_URL = "http://localhost:8080/"
    # otherwise
    BASE_URL = "http://linnet.doc.ic.ac.uk:8000/"
    RECOMPUTE_EVERY = 20
    USE_FAST_CORRESPONDENCES = False
    DRAW_CORRESPONDENCES = False
    TRIES_LIMIT = 100
    RUN_LOCALLY = True

    NUM_PAIRS = 8
    LOAD_SIZE = 224
    LAYER = 9
    FACET = "key"
    BIN = True
    THRESH = 0.05
    MODEL_TYPE = "dino_vits8"
    STRIDE = 4
    PATCH_SIZE = 8
    DEVICE = "cuda"

    # Simulation constants
    USE_GUI = True
    VERBOSITY = 1
    GRAVITY = -9.81
    ARM_BASE_POSITION = (0, 0, 1.2)
    TABLE_BASE_POSITION = (0, 0, 0)
    # move this more to the front of the robot
    ARM_HOME_POSITION = (0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0, 0, 0, 0, 0)
    OBJECT_X_Y_Z_BASE = (0.4, 0.0, 1.3)
    DEBUG_CAMERA_VIEW = (3, 45, -40)
    MOVE_TO_TARGET_ERROR_THRESHOLD = 0.01
    CAMERA_FOV = 60
    CAMERA_NEAR_PLANE = 0.01
    CAMERA_FAR_PLANE = 100
    CAMERA_INIT_VECTOR = (0, 0, 1)
    CAMERA_INIT_UP = (0, 1, 0)
    CAMERA_TO_EEF_TRANSLATION = (0, 0, 0.05)
    CAMERA_TO_EEF_ROTATION = (0, 0, -np.pi / 2)
    TAKE_IMAGE_AT_EVERY_STEP = False
    RANDOM_OBJECT_POSITION = True
    RANDOM_OBJECT_POSITION_FOLLOWING = False
    RANDOM_OBJECT_ROTATION = True
    MAX_STEPS = 200

    def get_dino_config(self):
        return {
            "num_pairs": self.NUM_PAIRS,
            "load_size": self.LOAD_SIZE,
            "layer": self.LAYER,
            "facet": self.FACET,
            "bin": self.BIN,
            "thresh": self.THRESH,
            "model_type": self.MODEL_TYPE,
            "stride": self.STRIDE,
            "device": self.DEVICE,
            "draw": self.DRAW_CORRESPONDENCES,
            "run_locally": self.RUN_LOCALLY,
        }
