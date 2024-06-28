class Config:
    IMAGE_SIZE = 224
    ERR_THRESHOLD = 0.05
    IMAGE_DIR = "images/"
    # using the ssh tunnel from home ssh -L 8080:linnet.doc.ic.ac.uk:8000 md1823@shell5.doc.ic.ac.uk
    BASE_URL = "http://localhost:8080/"
    # otherwise
    # BASE_URL = 'http://linnet.doc.ic.ac.uk:8000/'

    VERBOSITY = 1
