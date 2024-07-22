import os

from config import Config
from database import create_and_populate_db
from demo_sim_env import DemoSimEnv
from dinobot import run_dino_once
from task_types import Task


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


def run_self_experiment(task):
    config = Config()
    config.VERBOSITY = 0
    config.USE_FAST_CORRESPONDENCES = True
    config.USE_GUI = False
    config.RUN_LOCALLY = False
    db = create_and_populate_db(config)

    names = db.get_all_object_names_for_task(task)

    print(f"Running self transfer experiment for task {task}")

    num_tries = 10
    for name in names:
        all_tries = []
        success_count = 0
        for s in range(find_first(name, name, config, num_tries), num_tries):
            config.SEED = s
            success, tries = run_dino_once(config, db, name, name, task)
            if success:
                success_count += 1
            all_tries.append(tries)
        if len(all_tries) == 0:
            print(f"Skipped {name}->{name}")
            continue
        print(
            f"For transfer {name}->{name}: {success_count}/{num_tries} success rate with "
            f"{sum(all_tries) / len(all_tries)} steps on average"
        )


def run_cross_experiment(task):
    config = Config()
    config.VERBOSITY = 0
    config.USE_FAST_CORRESPONDENCES = True
    config.USE_GUI = False
    config.RUN_LOCALLY = False
    db = create_and_populate_db(config)

    names = db.get_all_object_names_for_task(task)

    print(f"Running cross transfer experiment for task {task}")

    num_tries = 10
    for base in names:
        for target in names:
            all_tries = []
            success_count = 0
            for s in range(find_first(base, target, config, num_tries), num_tries):
                config.SEED = s
                success, tries = run_dino_once(config, db, base, target, task)
                if success:
                    success_count += 1
                all_tries.append(tries)
            if len(all_tries) == 0:
                print(f"Skipped {base}->{target}")
                continue
            print(
                f"For transfer {base}->{target}: {success_count}/{num_tries} success rate with "
                f"{sum(all_tries) / len(all_tries)} steps on average"
            )


def replay_transfer(base_object, target_object, num, task):
    config = Config()
    db = create_and_populate_db(config)
    config.VERBOSITY = 1
    sim = DemoSimEnv(config, task, *db.get_load_info(target_object, task))
    sim.load_state(
        f"_generated/transfers/transfer_{base_object}_{target_object}_{str(num).zfill(3)}.json"
    )
    demo = sim.load_demonstration(db.get_demo_for_object(base_object))["demo_positions"]
    success = sim.replay_demo(demo)
    sim.disconnect()
    print(success)


if __name__ == "__main__":
    run_cross_experiment(Task.GRASPING.value)
