from config import Config
from database import DB
from dinobot import run_dino_once

if __name__ == "__main__":
    config = Config()
    config.VERBOSITY = 0
    config.USE_FAST_CORRESPONDENCES = True
    config.USE_GUI = False
    db = DB(config)

    names = db.get_all_object_names()

    num_tries = 10

    for n in names:
        success_count = 0
        for s in range(num_tries):
            config.SEED = s
            success = run_dino_once(config, db, n, n)
            print(n, s, success)
            if success:
                success_count += 1
        print(f"For object {n}: {success_count}/{num_tries} success rate")
