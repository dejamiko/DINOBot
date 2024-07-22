import json
import os

from task_types import Task


def update_old_jsons():
    # load the state to json
    # add task: "grasping"
    # save to json as previously
    base_dir = "_generated/transfers/grasping_old/"
    for state in os.listdir(base_dir):
        if not state.endswith(".json"):
            continue
        with open(os.path.join(base_dir, state)) as f:
            data = json.load(f)
        data["task"] = Task.GRASPING.value

        with open(os.path.join(base_dir, "../new/", state), "w") as f:
            json.dump(data, f)


if __name__ == "__main__":
    pass
