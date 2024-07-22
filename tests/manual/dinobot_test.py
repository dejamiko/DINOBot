from config import Config
from database import DB
from dinobot import run_dino_once
from task_types import Task

if __name__ == "__main__":
    # note, for paths to work, this needs to be run with the working directory being the parent DINOBot directory
    config = Config()
    config.LOAD_SIZE = 80
    config.USE_FAST_CORRESPONDENCES = True
    config.RUN_LOCALLY = True
    config.VERBOSITY = 1
    config.USE_GUI = True
    config.DEVICE = "cpu"

    db = DB(config)

    name = db.get_all_object_names_for_task(Task.GRASPING.value)[0]
    success = run_dino_once(config, db, name, name, Task.GRASPING.value)
    print(f"For object {name} achieved success: {success}")
