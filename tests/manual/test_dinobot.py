from config import Config

if __name__ == "__main__":
    # note, for paths to work, this needs to be run with the working directory being the parent DINOBot directory
    config = Config()
    config.LOAD_SIZE = 40
    config.USE_FAST_CORRESPONDENCES = True
    config.RUN_LOCALLY = True
    config.VERBOSITY = 1
    config.USE_GUI = True
    config.DEVICE = "cpu"

    # TODO perhaps mock this somehow for the purpose of easy testing
    demo_path = "demonstrations/demonstration_001.json"
