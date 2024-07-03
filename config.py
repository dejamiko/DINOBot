class Config:
    ERR_THRESHOLD = 0.01
    IMAGE_DIR = "images/"
    # using the ssh tunnel from home ssh -L 8080:linnet.doc.ic.ac.uk:8000 md1823@shell5.doc.ic.ac.uk
    BASE_URL = "http://localhost:8080/"
    # otherwise
    # BASE_URL = 'http://linnet.doc.ic.ac.uk:8000/'

    VERBOSITY = 1

    MAX_ACTION = 0.05

    num_pairs = 8  # @param
    load_size = 480  # @param
    layer = 9  # @param
    facet = 'key'  # @param
    bin = True  # @param
    thresh = 0.2  # @param
    model_type = 'dino_vits8'  # @param
    stride = 8  # @param
    patch_size = 8  # not changeable
    device = "cpu"