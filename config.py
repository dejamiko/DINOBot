class Config:
    # DINOBot deployment constants
    ERR_THRESHOLD = 0.04
    IMAGE_DIR = "images/"
    # using the ssh tunnel from home ssh -L 8080:linnet.doc.ic.ac.uk:8000 md1823@shell5.doc.ic.ac.uk
    BASE_URL = "http://localhost:8080/"
    # otherwise
    # BASE_URL = 'http://linnet.doc.ic.ac.uk:8000/'
    RECOMPUTE_EVERY = 20
    USE_FAST_CORRESPONDENCES = False
    DRAW_CORRESPONDENCES = False

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
    VERBOSITY = 1

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
        }
