import json
import os
import re
import shutil
import sqlite3

import cv2
import numpy as np
import pybullet_data
from pybullet_object_models import ycb_objects

from config import Config
from task_types import Task


def get_path(source):
    if source == "pybullet_data":
        return pybullet_data.getDataPath()
    elif source == "ycb":
        return ycb_objects.getDataPath()
    elif source == "test":
        print("This should only be used for tests")
        return "tests/"
    raise ValueError(f"Unknown source '{source}'")


class DB:
    def __init__(self, config, name="dino.db"):
        self.config = config
        self.con = sqlite3.connect(os.path.join(config.BASE_DIR, name))

    def create_tables(self):
        sql_string = (
            "CREATE TABLE objects"
            "("
            "object_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "object_name VARCHAR(50) UNIQUE NOT NULL, "
            "urdf_path VARCHAR(100) UNIQUE NOT NULL, "
            "source VARCHAR(20) NOT NULL"
            ")"
        )
        self._create_table("objects", sql_string)

        sql_string = (
            "CREATE TABLE transfers"
            "("
            "object_id_1 INT NOT NULL,"
            "object_id_2 INT NOT NULL,"
            "task VARCHAR(20) NOT NULL,"
            "success_rate FLOAT NOT NULL,"
            "FOREIGN KEY (object_id_1) REFERENCES objects(object_id),"
            "FOREIGN KEY (object_id_2) REFERENCES objects(object_id)"
            ")"
        )
        self._create_table("transfers", sql_string)

        sql_string = (
            "CREATE TABLE demonstrations"
            "("
            "object_id INT NOT NULL,"
            "task VARCHAR(20) NOT NULL,"
            "demo_path VARCHAR(100) UNIQUE NOT NULL, "
            "scale FLOAT,"
            "pos_x FLOAT,"
            "pos_y FLOAT,"
            "pos_z FLOAT,"
            "rot_r FLOAT,"
            "rot_p FLOAT,"
            "rot_y FLOAT,"
            "rot_r_adj FLOAT,"
            "rot_p_adj FLOAT,"
            "rot_y_adj FLOAT,"
            "FOREIGN KEY (object_id) REFERENCES objects(object_id),"
            "UNIQUE (object_id, task)"
            ")"
        )
        self._create_table("demonstrations", sql_string)

    def _create_table(self, table_name, sql_string):
        try:
            self.con.execute(sql_string)
        except sqlite3.OperationalError as e:
            if str(e) == f"table {table_name} already exists":
                pass
            else:
                raise e

    @staticmethod
    def _check_path(path, extension):
        assert os.path.isfile(path), f"{path} is not a file"
        assert path.endswith(extension), f"{path} is not of type {extension}"

    def remove_all_tables(self):
        with self.con:
            for table in ["demonstrations", "transfers", "objects"]:
                try:
                    self.con.execute(f"DROP TABLE {table}")
                except sqlite3.OperationalError as e:
                    if str(e).startswith("no such table: "):
                        pass
                    else:
                        raise e

    def get_demo(self, object_name, task):
        object_id = self._get_object_id_by_name(object_name)

        assert (
                object_id is not None
        ), f"There is no object {object_name} in the database"

        res = self.con.execute(
            f"SELECT demo_path FROM demonstrations WHERE object_id=? AND task=?",
            (object_id, task),
        ).fetchone()
        return res[0] if res is not None else None

    def get_load_info(self, object_name, task):
        object_id = self._get_object_id_by_name(object_name)

        assert (
                object_id is not None
        ), f"There is no object {object_name} in the database"

        (
            scale,
            pos_x,
            pos_y,
            pos_z,
            rot_r,
            rot_p,
            rot_y,
            rot_r_adj,
            rot_p_adj,
            rot_y_adj,
        ) = self.con.execute(
            (
                f"SELECT scale, pos_x, pos_y, pos_z, rot_r, rot_p, rot_y, rot_r_adj, rot_p_adj, rot_y_adj "
                f"FROM demonstrations WHERE object_id=? AND task=?"
            ),
            (object_id, task),
        ).fetchone()

        urdf_path, source = self.con.execute(
            f"SELECT urdf_path, source FROM objects WHERE object_id=?", (object_id,)
        ).fetchone()

        load_path = os.path.join(get_path(source), urdf_path)

        # provide a default value (is there a better way for this?)
        scale = 1.0 if scale is None else scale
        pos_x = 0.0 if pos_x is None else pos_x
        pos_y = 0.0 if pos_y is None else pos_y
        pos_z = 0.0 if pos_z is None else pos_z
        rot_r = 0.0 if rot_r is None else rot_r
        rot_p = 0.0 if rot_p is None else rot_p
        rot_y = 0.0 if rot_y is None else rot_y
        rot_r_adj = 0.0 if rot_r_adj is None else rot_r_adj
        rot_p_adj = 0.0 if rot_p_adj is None else rot_p_adj
        rot_y_adj = 0.0 if rot_y_adj is None else rot_y_adj

        return (
            load_path,
            scale,
            (pos_x, pos_y, pos_z),
            (rot_r, rot_p, rot_y),
            (rot_r_adj, rot_p_adj, rot_y_adj),
        )

    def get_success_rate(self, object_name_1, object_name_2, task):
        object_id_1 = self._get_object_id_by_name(object_name_1)
        object_id_2 = self._get_object_id_by_name(object_name_2)

        assert (
                object_id_1 is not None
        ), f"There is no object {object_name_1} in the database"
        assert (
                object_id_2 is not None
        ), f"There is no object {object_name_2} in the database"

        success_rate = self.con.execute(
            f"SELECT success_rate FROM transfers WHERE object_id_1=? AND object_id_2=? AND task=?",
            (object_id_1, object_id_2, task),
        ).fetchone()

        return success_rate[0] if success_rate is not None else None

    def get_all_object_names_for_task(self, task):
        res = self.con.execute(
            f"SELECT object_name "
            f"FROM (objects JOIN demonstrations ON objects.object_id=demonstrations.object_id)"
            f"WHERE task=?"
            f"ORDER BY object_name",
            (task,),
        )
        return [x[0] for x in res.fetchall()]

    def add_object(self, object_name, urdf_path, source):
        self._check_path(os.path.join(get_path(source), urdf_path), ".urdf")
        prev = self._get_object_id_by_name(object_name)

        with self.con as c:
            if prev is not None:
                c.execute(
                    f"UPDATE objects SET urdf_path=?, source=? WHERE object_id=?",
                    (urdf_path, source, prev),
                )
            else:
                c.execute(
                    f"INSERT INTO objects VALUES(?,?,?,?)",
                    (None, object_name, urdf_path, source),
                )

    def add_demo(
            self, object_name, task, demo_path, scale=None, pos=None, rot=None, rot_adj=None
    ):
        self._check_path(demo_path, ".json")

        object_id = self._get_object_id_by_name(object_name)
        assert (
                object_id is not None
        ), f"There is no object {object_name} in the database"

        prev = self.get_demo(object_name, task)

        if pos is not None:
            x, y, z = pos
        else:
            x, y, z = None, None, None

        if rot is not None:
            r, p, yaw = rot
        else:
            r, p, yaw = None, None, None

        if rot_adj is not None:
            r_adj, p_adj, yaw_adj = rot_adj
        else:
            r_adj, p_adj, yaw_adj = None, None, None

        with self.con as c:
            if prev is not None:
                c.execute(
                    f"UPDATE demonstrations "
                    f"SET demo_path=?, scale=?, pos_x=?, pos_y=?, pos_z=?, rot_r=?, rot_p=?, rot_y=?, "
                    f"rot_r_adj=?, rot_p_adj=?, rot_y_adj=? "
                    f"WHERE object_id=? AND task=?",
                    (
                        demo_path,
                        scale,
                        x,
                        y,
                        z,
                        r,
                        p,
                        yaw,
                        r_adj,
                        p_adj,
                        yaw_adj,
                        object_id,
                        task,
                    ),
                )
            else:
                c.execute(
                    f"INSERT INTO demonstrations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        object_id,
                        task,
                        demo_path,
                        scale,
                        x,
                        y,
                        z,
                        r,
                        p,
                        yaw,
                        r_adj,
                        p_adj,
                        yaw_adj,
                    ),
                )

    def add_transfer(self, object_name_1, object_name_2, task, success_rate):
        object_id_1 = self._get_object_id_by_name(object_name_1)
        object_id_2 = self._get_object_id_by_name(object_name_2)

        assert (
                object_id_1 is not None
        ), f"There is no object {object_name_1} in the database"
        assert (
                object_id_2 is not None
        ), f"There is no object {object_name_2} in the database"

        assert (
                0 <= success_rate <= 1
        ), f"success rate should be between 0 and 1, got {success_rate}"

        prev = self.get_success_rate(object_name_1, object_name_2, task)

        with self.con as c:
            if prev is not None:
                c.execute(
                    f"UPDATE transfers "
                    f"SET success_rate=? "
                    f"WHERE object_id_1=? AND object_id_2=? AND task=?",
                    (success_rate, object_id_1, object_id_2, task),
                )
            else:
                c.execute(
                    f"INSERT INTO transfers VALUES(?,?,?,?)",
                    (object_id_1, object_id_2, task, success_rate),
                )

    def _get_object_id_by_name(self, object_name):
        res = self.con.execute(
            f"SELECT object_id FROM objects WHERE object_name=?", (object_name,)
        )
        res = res.fetchone()

        return res[0] if res is not None else res

    def get_nail_object(self):
        # TODO this is hard-coded and not nice at all
        return self.get_load_info("chips_can", Task.GRASPING.value)[0]

    def export_data_for_learning(self):
        export_path = os.path.join(self.config.BASE_DIR, "export")
        self._clear_directory(export_path)
        # create a json file with the structure (obj_name, task): path_to_images
        object_data = {}
        viable_objects = get_viable_objects(db)
        obj_task_list = []
        for t in viable_objects.keys():
            for n in viable_objects[t]:
                demo_path = self.get_demo(n, t)
                image_path, image_embeddings = self._save_images_and_get_embeddings(demo_path, export_path, n, t)
                object_data[f"{n}-{t}"] = {
                    "image_path": image_path,
                    "image_embeddings": image_embeddings
                }
                obj_task_list.append((n, t))

        with open(os.path.join(export_path, "objects.json"), "w") as f:
            json.dump(object_data, f)

        transfers = {}
        for t in viable_objects.keys():
            for n in viable_objects[t]:
                for n2 in viable_objects[t]:
                    transfers[f"{n}-{n2}-{t}"] = self.get_success_rate(n, n2, t)
        with open(os.path.join(export_path, "transfers.json"), "w") as f:
            json.dump(transfers, f)

    @staticmethod
    def _clear_directory(export_path):
        if os.path.isdir(export_path) and len(os.listdir(export_path)) != 0:
            for filename in os.listdir(export_path):
                file_path = os.path.join(export_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print("Failed to delete %s. Reason: %s" % (file_path, e))

    def _save_images_and_get_embeddings(self, demo_path, export_path, obj_name, task):
        from demo_sim_env import DemoSimEnv
        from DINOserver.client import get_embeddings

        new_dir_path = os.path.join(export_path, f"{obj_name}_{task}/")
        os.makedirs(new_dir_path)
        data = DemoSimEnv.load_demonstration(demo_path)
        images = data["images"]
        image_embeddings = []
        for i, im in enumerate(images):
            img_path = os.path.join(new_dir_path, f"image_{i}.png")
            cv2.imwrite(img_path, im)
            image_embeddings.append(get_embeddings(img_path, self.config.BASE_URL))

        return new_dir_path, image_embeddings


ycb_names = [
    "YcbBanana",
    "YcbChipsCan",
    "YcbCrackerBox",
    "YcbGelatinBox",
    "YcbHammer",
    "YcbMasterChefCan",
    "YcbMediumClamp",
    "YcbMustardBottle",
    "YcbPear",
    "YcbPottedMeatCan",
    "YcbPowerDrill",
    "YcbScissors",
    "YcbTomatoSoupCan",
]

urdf_paths = {
    "bike": "bicycle/bike.urdf",
    "domino": "domino/domino.urdf",
    "jenga": "jenga/jenga.urdf",
    "mini_cheetah": "mini_cheetah/mini_cheetah.urdf",
    "minitaur": "quadruped/minitaur.urdf",
    "mug": "urdf/mug.urdf",
    "007": "random_urdfs/007/007.urdf",
    "014": "random_urdfs/014/014.urdf",
    "024": "random_urdfs/024/024.urdf",
    "033": "random_urdfs/033/033.urdf",
    "088": "random_urdfs/088/088.urdf",
    "117": "random_urdfs/117/117.urdf",
    "119": "random_urdfs/119/119.urdf",
    "132": "random_urdfs/132/132.urdf",
    "133": "random_urdfs/133/133.urdf",
    "184": "random_urdfs/184/184.urdf",
    "185": "random_urdfs/185/185.urdf",
    "227": "random_urdfs/227/227.urdf",
    "228": "random_urdfs/228/228.urdf",
    "238": "random_urdfs/238/238.urdf",
    "000": "random_urdfs/000/000.urdf",
    "011": "random_urdfs/011/011.urdf",
    "016": "random_urdfs/016/016.urdf",
    "020": "random_urdfs/020/020.urdf",
    "052": "random_urdfs/052/052.urdf",
    "057": "random_urdfs/057/057.urdf",
    "058": "random_urdfs/058/058.urdf",
    "078": "random_urdfs/078/078.urdf",
    "080": "random_urdfs/080/080.urdf",
    "081": "random_urdfs/081/081.urdf",
    "083": "random_urdfs/083/083.urdf",
    "090": "random_urdfs/090/090.urdf",
    "114": "random_urdfs/114/114.urdf",
    "115": "random_urdfs/115/115.urdf",
    "126": "random_urdfs/126/126.urdf",
    "147": "random_urdfs/147/147.urdf",
    "157": "random_urdfs/157/157.urdf",
    "001": "random_urdfs/001/001.urdf",
    "003": "random_urdfs/003/003.urdf",
    "005": "random_urdfs/005/005.urdf",
    "008": "random_urdfs/008/008.urdf",
    "009": "random_urdfs/009/009.urdf",
    "010": "random_urdfs/010/010.urdf",
    "012": "random_urdfs/012/012.urdf",
    "015": "random_urdfs/015/015.urdf",
    "018": "random_urdfs/018/018.urdf",
    "019": "random_urdfs/019/019.urdf",
    "021": "random_urdfs/021/021.urdf",
    "023": "random_urdfs/023/023.urdf",
    "025": "random_urdfs/025/025.urdf",
    "026": "random_urdfs/026/026.urdf",
    "027": "random_urdfs/027/027.urdf",
    "029": "random_urdfs/029/029.urdf",
    "030": "random_urdfs/030/030.urdf",
    "034": "random_urdfs/034/034.urdf",
    "035": "random_urdfs/035/035.urdf",
    "037": "random_urdfs/037/037.urdf",
}

scales = {
    Task.GRASPING.value: {
        "chips_can": 0.7,
        "cracker_box": 0.7,
        "gelatin_box": 1.5,
        "bike": 0.05,
        "domino": 2.5,
        "mini_cheetah": 0.25,
        "minitaur": 0.25,
        "mug": 0.8,
    },
    Task.PUSHING.value: {
        "024": 1.2,
        "185": 1.5,
        "227": 1.5,
        "228": 1.5,
        "238": 1.5,
        "cracker_box": 0.8,
        "gelatin_box": 1.5,
    },
    Task.HAMMERING.value: {
        "016": 1.3,
        "024": 1.2,
        "052": 1.2,
        "078": 1.2,
        "081": 1.2,
        "126": 1.2,
        "147": 1.1,
        "157": 1.2,
    },
    Task.GRASPING_SIMP.value: {}
}

positions = {
    Task.GRASPING.value: {},
    Task.PUSHING.value: {},
    Task.HAMMERING.value: {},
    Task.GRASPING_SIMP.value: {
        "015": (0, -0.05, 0),
        "024": (0, -0.04, 0),
    }
}

rotations = {
    Task.GRASPING.value: {
        "mini_cheetah": (0, np.pi, 0),
    },
    Task.PUSHING.value: {},
    Task.HAMMERING.value: {
        "hammer": (0, 0, 2 * np.pi / 4),
        "000": (0, 0, 6 * np.pi / 4),
        "007": (0, 0, 2 * np.pi / 4),
        "011": (0, 0, 6 * np.pi / 4),
        "016": (0, 0, 2 * np.pi / 4),
        "020": (0, 0, 6 * np.pi / 4),
        "024": (0, 0, 6 * np.pi / 4),
        "052": (0, 0, 2 * np.pi / 4),
        "057": (0, 0, 6 * np.pi / 4),
        "058": (0, 0, 6 * np.pi / 4),
        "078": (0, 0, 2 * np.pi / 4),
        "080": (0, 0, 2 * np.pi / 4),
        "081": (0, 0, 2 * np.pi / 4),
        "083": (0, 0, 2 * np.pi / 4),
        "090": (0, 0, 2 * np.pi / 4),
        "114": (0, 0, 6 * np.pi / 4),
        "115": (0, 0, 6 * np.pi / 4),
        "126": (0, 0, 6 * np.pi / 4),
        "147": (0, 0, 2 * np.pi / 4),
        "157": (0, 0, 6 * np.pi / 4),
    },
    Task.GRASPING_SIMP.value: {
        "000": (0, 0, 3 * np.pi / 2),
        "001": (0, 0, np.pi / 2),
        "003": (0, np.pi / 2, np.pi / 2),
        "005": (0, 0, np.pi / 2),
        "007": (0, 0, np.pi / 2),
        "008": (0, 0, np.pi / 2),
        "009": (0, 0, np.pi / 2),
        "010": (0, 0, np.pi / 2),
        "011": (0, 0, 3 * np.pi / 2),
        "012": (0, 0, 3 * np.pi / 2),
        "014": (0, 0, 3 * np.pi / 2),
        "015": (0, 0, np.pi / 2),
        "016": (0, 0, np.pi / 2),
        "018": (0, 0, np.pi / 2),
        "019": (0, 0, 3 * np.pi / 2),
        "020": (0, 0, 3 * np.pi / 2),
        "021": (0, np.pi / 2, np.pi / 2),
        "023": (0, np.pi / 2, np.pi / 2),
        "024": (0, np.pi / 2, 3 * np.pi / 2),
        "025": (0, np.pi / 2, 3 * np.pi / 2),
        "026": (0, np.pi / 2, np.pi / 2),
        "027": (0, 0, np.pi / 2),
        "029": (0, 0, 3 * np.pi / 2),
        "030": (0, 0, np.pi / 2),
        "033": (0, 0, np.pi / 2),
        "034": (0, 0, np.pi / 2),
        "035": (0, 0, np.pi / 2),
        "037": (np.pi / 2, 0, np.pi / 2),
    }
}

adj_rotations = {
    Task.GRASPING.value: {},
    Task.PUSHING.value: {
        "banana": (0, 0, 3 * np.pi / 2),
        "gelatin_box": (0, 0, 3 * np.pi / 4),
        "power_drill": (0, 0, 3 * np.pi / 4),
    },
    Task.HAMMERING.value: {
        "000": (0, 0, np.pi),
        "011": (0, 0, np.pi),
        "020": (0, 0, np.pi),
        "024": (0, 0, np.pi),
        "057": (0, 0, np.pi),
        "058": (0, 0, np.pi),
        "114": (0, 0, np.pi),
        "115": (0, 0, np.pi),
        "126": (0, 0, np.pi),
        "157": (0, 0, np.pi),
    },
    Task.GRASPING_SIMP.value: {}
}


def snake_to_pascal_case(name):
    return name.replace("_", " ").title().replace(" ", "")


def pascal_to_snake_case(name):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def populate_transfers(db, task):
    reg = re.compile(
        r"For transfer ([a-z0-9_]*)->([a-z0-9_]*): (\d*)/10 success rate with -?\d*.?\d* steps on average"
    )
    with open(f"_generated/{task}_cross_experiment") as f:
        lines = f.readlines()
        for l in lines:
            m = reg.match(l)
            base, target, num = m.groups()
            db.add_transfer(base, target, task, int(num) / 10.0)

    if not os.path.exists(f"_generated/{task}_cross_experiment_replays"):
        return
    with open(f"_generated/{task}_cross_experiment_replays") as f:
        lines = f.readlines()
        for l in lines:
            m = reg.match(l)
            base, target, num = m.groups()
            num = int(num) / 10.0
            num2 = db.get_success_rate(base, target, task)
            if num != num2:
                if num2 is None:
                    db.add_transfer(base, target, task, num)
                elif base == target:
                    db.add_transfer(base, target, task, max(num, num2))
                else:
                    db.add_transfer(base, target, task, min(num, num2))


def create_and_populate_db(config):
    db = DB(config)
    db.remove_all_tables()
    db.create_tables()
    base_dir = "demonstrations/"
    sub_dirs = [x[0] for x in os.walk(base_dir)]
    for directory in sub_dirs:
        for f in os.listdir(directory):
            if not f.endswith(".json"):
                continue
            object_name = "_".join(f.split("_")[1:])[:-5]
            task = directory.split("/")[-1]
            is_ycb = f"Ycb{snake_to_pascal_case(object_name)}" in ycb_names
            if is_ycb:
                urdf_path = f"Ycb{snake_to_pascal_case(object_name)}/model.urdf"
                source = "ycb"
            else:
                urdf_path = urdf_paths[object_name]
                source = "pybullet_data"
            db.add_object(object_name, urdf_path, source)
            db.add_demo(
                object_name,
                task,
                os.path.join(directory, f),
                scales[task].get(object_name, None),
                positions[task].get(object_name, None),
                rotations[task].get(object_name, None),
                adj_rotations[task].get(object_name, None),
            )
    for task in Task:
        populate_transfers(db, task.value)
    return db


def get_num_of_successes(name, names, task, db):
    successes = 0
    for n in names:
        if n == name:
            continue
        s_r = db.get_success_rate(name, n, task)
        if s_r >= 0.7:
            successes += 1
    return successes


def get_self_success(n, task, db):
    return db.get_success_rate(n, n, task) >= 0.7


def get_viable_objects(db):
    viable_names = {}
    for task in Task:
        names = db.get_all_object_names_for_task(task.value)
        names_to_remove = set()
        for n in names:
            if get_num_of_successes(n, names, task.value, db) >= 10:
                print(n, task.value, get_num_of_successes(n, names, task.value, db))
                names_to_remove.add(n)
            if not get_self_success(n, task.value, db):
                print(f"No self success for {n}, {task.value}")
                names_to_remove.add(n)
        viable_names[task.value] = sorted(list(set(names) - names_to_remove))
    return viable_names


if __name__ == "__main__":
    config = Config()
    db = create_and_populate_db(config)
    print(db.get_all_object_names_for_task(Task.GRASPING.value))
    print(db.get_all_object_names_for_task(Task.PUSHING.value))
    print(db.get_all_object_names_for_task(Task.HAMMERING.value))
    print(db.get_all_object_names_for_task(Task.GRASPING_SIMP.value))
    print(db.get_demo("banana", Task.GRASPING.value))
    print(db.get_demo("banana", Task.PUSHING.value))
    print(db.get_demo("hammer", Task.HAMMERING.value))
    print(db.get_demo("018", Task.GRASPING_SIMP.value))

    # db.export_data_for_learning()
