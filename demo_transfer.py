import os

from config import Config
from database import create_and_populate_db
from demo_sim_env import DemoSimEnv
from dinobot import run_dino_once


def run_self_experiment(names):
    config = Config()
    config.VERBOSITY = 0
    config.USE_FAST_CORRESPONDENCES = True
    config.USE_GUI = False
    config.RUN_LOCALLY = False
    db = create_and_populate_db(config)

    num_tries = 10
    for n in names:
        success_count = 0
        for s in range(num_tries):
            config.SEED = s
            success, tries = run_dino_once(config, db, n, n)
            print(n, s, success, tries)
            if success:
                success_count += 1
        print(f"For object {n}: {success_count}/{num_tries} success rate")


def find_first(base, target, config, num_tries):
    i = 0
    while i < num_tries:
        filename = os.path.join(
            config.BASE_DIR,
            "transfers",
            f"transfer_{base}_{target}_{str(i).zfill(3)}.json",
        )
        if not os.path.exists(filename):
            break
        i += 1
    return i


def run_cross_experiment():
    config = Config()
    config.VERBOSITY = 0
    config.USE_FAST_CORRESPONDENCES = True
    config.USE_GUI = False
    config.RUN_LOCALLY = True
    db = create_and_populate_db(config)

    names = db.get_all_object_names()

    num_tries = 10
    for base in names:
        for target in names:
            success_count = 0
            for s in range(find_first(base, target, config, num_tries), num_tries):
                config.SEED = s
                success, tries = run_dino_once(
                    config, db, base_object=base, target_object=target
                )
                if success:
                    success_count += 1
            print(
                f"For transfer {base}->{target}: {success_count}/{num_tries} success rate"
            )


def replay_transfer(base_object, target_object, num):
    config = Config()
    db = create_and_populate_db(config)
    config.VERBOSITY = 1
    sim = DemoSimEnv(
        config, db.get_urdf_path(target_object), db.get_urdf_scale(target_object)
    )
    sim.load_state(
        f"_generated/transfers/transfer_{base_object}_{target_object}_{str(num).zfill(3)}.json"
    )
    demo = sim.load_demonstration(db.get_demo_for_object(base_object))[
        "demo_velocities"
    ]
    success = sim.replay_demo(demo)
    sim.disconnect()
    print(success)


if __name__ == "__main__":
    run_cross_experiment()
