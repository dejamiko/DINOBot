import numpy as np


class Config:
    SEED = 0
    BASE_DIR = "_generated/"

    # DINOBot deployment constants
    ERR_THRESHOLD = 0.07
    IMAGE_DIR = BASE_DIR + "images/"
    # using the ssh tunnel from home ssh -L 8080:linnet.doc.ic.ac.uk:8000 md1823@shell5.doc.ic.ac.uk
    # BASE_URL = "http://localhost:8080/"
    BASE_URL = "http://linnet.doc.ic.ac.uk:8000/"  # otherwise
    RECOMPUTE_EVERY = 100
    USE_FAST_CORRESPONDENCES = False
    DRAW_CORRESPONDENCES = False
    TRIES_LIMIT = 25
    RUN_LOCALLY = True

    # DINO constants
    NUM_PAIRS = 8
    LOAD_SIZE = 400
    LAYER = 9
    FACET = "key"
    BIN = True
    THRESH = 0.05
    MODEL_TYPE = "dino_vits8"
    STRIDE = 8
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
    MOVE_TO_TARGET_ERROR_THRESHOLD = 1e-4
    CAMERA_FOV = 60
    CAMERA_NEAR_PLANE = 0.01
    CAMERA_FAR_PLANE = 100
    CAMERA_INIT_VECTOR = (0, 0, 1)
    CAMERA_INIT_UP = (0, 1, 0)
    CAMERA_TO_EEF_TRANSLATION = (0, 0, 0.02)
    CAMERA_TO_EEF_ROTATION = (0, 0, np.pi / 2)
    TAKE_IMAGE_AT_EVERY_STEP = False
    RANDOM_OBJECT_POSITION = True
    RANDOM_OBJECT_POSITION_FOLLOWING = False
    RANDOM_OBJECT_ROTATION = True
    MOVEMENT_ITERATION_MAX_STEPS = 100

    DEMO_ADDITIONAL_IMAGE_ANGLE = 0.174533  # 10 degrees in radians

    GRASP_SUCCESS_HEIGHT = 0.2
    PUSH_SUCCESS_DIST = 0.3
    PUSH_SUCCESS_ANGLE = 0.436332  # 45 degrees in radians

    HAMMERING_ADDITIONAL_OBJECT_OFFSET = (0.15, 0, 0)
    HAMMERING_ADDITIONAL_OBJECT_ROTATION = (0, 0, 0)
    HAMMERING_ADDITIONAL_OBJECT_SCALE = 0.5

    HAMMERING_SUCCESS_FORCE = 200
    HAMMERING_SUCCESS_ALIGNMENT = 0.95

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
