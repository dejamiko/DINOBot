class Config:
    ERR_THRESHOLD = 0.04
    IMAGE_DIR = "images/"
    # using the ssh tunnel from home ssh -L 8080:linnet.doc.ic.ac.uk:8000 md1823@shell5.doc.ic.ac.uk
    BASE_URL = "http://localhost:8080/"
    # otherwise
    # BASE_URL = 'http://linnet.doc.ic.ac.uk:8000/'

    VERBOSITY = 1

    num_pairs = 8  # @param
    load_size = 224  # @param
    layer = 9  # @param
    facet = "key"  # @param
    bin = True  # @param
    thresh = 0.05  # @param
    model_type = "dino_vits8"  # @param
    stride = 4  # @param
    patch_size = 8  # not changeable
    device = "cuda"
    draw_correspondences = False

    RECOMPUTE_EVERY = 20
    USE_FAST_CORRESPONDENCES = False

    def get_dino_config(self):
        return {
            "num_pairs": self.num_pairs,
            "load_size": self.load_size,
            "layer": self.layer,
            "facet": self.facet,
            "bin": self.bin,
            "thresh": self.thresh,
            "model_type": self.model_type,
            "stride": self.stride,
            "device": self.device,
            "draw": self.draw_correspondences,
        }
