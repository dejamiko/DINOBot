import time

import wandb

from config import Config
from dinobot import run_dino_once


def run_fast_and_slow(config):
    config.VERBOSITY = 1
    config.USE_GUI = False
    config.USE_FAST_CORRESPONDENCES = True
    demo_path = "demonstrations/demonstration_001.json"
    num_of_runs = 10
    successes_1 = 0
    for i in range(num_of_runs):
        config.SEED = i
        success = run_dino_once(config, demo_path)
        if success:
            successes_1 += 1

    print(f"Successes fast {successes_1}")

    config.USE_FAST_CORRESPONDENCES = False
    successes_2 = 0
    for i in range(num_of_runs):
        config.SEED = i
        success = run_dino_once(config, demo_path)
        if success:
            successes_2 += 1

    print(f"Successes slow {successes_2}")

    return successes_1 + successes_2


def run_hyperparam_search():
    wandb.login(key="8d9dd70311672d46669adf913d75468f2ba2095b")

    sweep_config = {
        "name": "dinobot",
        "method": "bayes",
        "metric": {"name": "successes", "goal": "maximize"},
        "parameters": {
            "load_size": {"values": [160, 224, 240, 320, 360, 400]},
            "stride": {"values": [4, 8]},
            "thresh": {"min": 0.1, "max": 0.25},
            "err_threshold": {"min": 0.01, "max": 0.05},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="dinobot")

    def train(w_config=None):
        with wandb.init(config=w_config):
            w_config = wandb.config
            c = Config()

            c.LOAD_SIZE = w_config["load_size"]
            c.STRIDE = w_config["stride"]
            c.THRESH = w_config["thresh"]
            c.ERR_THRESHOLD = w_config["err_threshold"]

            wandb.log({"config": c.__dict__})
            print(f"Config: {c.__dict__}")

            start_time = time.time()

            successes = run_fast_and_slow(c)

            wandb.log({"successes": successes, "time_taken": time.time() - start_time})
            return successes

    wandb.agent(sweep_id, train, count=5)

    wandb.finish()


if __name__ == "__main__":
    run_hyperparam_search()
