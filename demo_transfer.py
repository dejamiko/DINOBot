import os
import time

from config import Config
from database import DB
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
    config.RUN_LOCALLY = True
    db = DB(config)

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
    config.RUN_LOCALLY = True
    db = DB(config)

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


def replay_transfer(config, db, base_object, target_object, num, task):
    sim = DemoSimEnv(config, task, *db.get_load_info(target_object, task))
    sim.load_state(
        f"_generated/transfers/{task}/transfer_{base_object}_{target_object}_{str(num).zfill(3)}.json"
    )
    demo = sim.load_demonstration(db.get_demo(base_object, task))["demo_positions"]
    success = sim.replay_demo(demo)
    sim.disconnect()
    return success


def ingest_transfers():
    config = Config()
    config.USE_GUI = False
    db = DB(config)
    config.VERBOSITY = 0
    results = {}
    base_dir = "_generated/transfers/grasping/"
    names = db.get_all_object_names_for_task("grasping")
    files = sorted(os.listdir(base_dir))
    start_time = time.time()
    prev = None
    for i, state in enumerate(files):
        if not state.endswith(".json"):
            continue
        state = state[:-5]
        num = state.split("_")[-1]
        earliest_occurrence = len(state)
        latest_occurrence = -1
        base = ""
        target = ""
        for n in names:
            if state.find(n) != -1 and state.find(n) < earliest_occurrence:
                earliest_occurrence = state.find(n)
                base = n
            if state.find(n) != -1 and state.find(n) > latest_occurrence:
                latest_occurrence = state.find(n)
                target = n
        num = int(num)
        success = replay_transfer(config, db, base, target, num, Task.GRASPING.value)
        if (base, target) not in results:
            results[(base, target)] = 0
        results[(base, target)] += success
        if prev is not None and (base, target) != prev:
            print(f"For transfer {base}->{target}: {results[(base, target)]}/10 success rate with -1 steps on average")
        prev = (base, target)
        if i % 10 == 0:
            print(
                f"Completed {i + 1}/{len(files)}, taking {time.time() - start_time} seconds"
            )


if __name__ == "__main__":
    # run_cross_experiment(Task.GRASPING.value)
    run_self_experiment(Task.PUSHING.value)
    # ingest_transfers()


