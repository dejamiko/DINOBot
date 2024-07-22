import json
import os

import numpy as np

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


def update_demos():
    base_dir = "demonstrations/grasping/"
    sub_dirs = [x[0] for x in os.walk(base_dir)]
    base_rot = np.array(
        [
            [9.99997288e-01, 1.21610064e-03, 1.98611110e-03],
            [1.21570485e-03, -9.99999241e-01, 2.00475162e-04],
            [1.98635339e-03, -1.98060093e-04, -9.99998008e-01],
        ]
    )
    for directory in sub_dirs:
        for f in os.listdir(directory):
            if not f.endswith(".json"):
                continue
            with open(os.path.join(directory, f), "r") as file:
                demonstration = json.load(file)
            new_recorded_data = []
            for p, r, g in demonstration["recorded_data"]:
                p = np.dot(np.linalg.inv(base_rot), p)
                new_recorded_data.append((p.tolist(), r, g))
            demonstration["recorded_data"] = new_recorded_data
            with open(os.path.join(directory, f), "w") as file:
                json.dump(demonstration, file)


if __name__ == "__main__":
    pass
