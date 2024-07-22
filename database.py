import os
import re
import sqlite3

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

        scale, pos_x, pos_y, pos_z, rot_r, rot_p, rot_y = self.con.execute(
            (
                f"SELECT scale, pos_x, pos_y, pos_z, rot_r, rot_p, rot_y "
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

        return load_path, scale, (pos_x, pos_y, pos_z), (rot_r, rot_p, rot_y)

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

    def add_demo(self, object_name, task, demo_path, scale=None, pos=None, rot=None):
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

        with self.con as c:
            if prev is not None:
                c.execute(
                    f"UPDATE demonstrations "
                    f"SET demo_path=?, scale=?, pos_x=?, pos_y=?, pos_z=?, rot_r=?, rot_p=?, rot_y=? "
                    f"WHERE object_id=? AND task=?",
                    (demo_path, scale, x, y, z, r, p, yaw, object_id, task),
                )
            else:
                c.execute(
                    f"INSERT INTO demonstrations VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (object_id, task, demo_path, scale, x, y, z, r, p, yaw),
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
    "duck": "duck_vhacd.urdf",
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
        "185": 1.5,
        "227": 1.5,
        "228": 1.5,
        "238": 1.5,
    },
    Task.HAMMERING.value: {},
}

positions = {
    Task.GRASPING.value: {},
    Task.PUSHING.value: {},
    Task.HAMMERING.value: {},
}

rotations = {
    Task.GRASPING.value: {
        "mini_cheetah": (0, np.pi, 0),
    },
    Task.PUSHING.value: {},
    Task.HAMMERING.value: {},
}


def snake_to_pascal_case(name):
    return name.replace("_", " ").title().replace(" ", "")


def pascal_to_snake_case(name):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def populate_transfers(db):
    reg = re.compile(
        r"For transfer ([a-z0-9_]*)->([a-z0-9_]*): (\d*)/10 success rate with -?\d*.?\d* steps on average"
    )
    with open("_generated/grasping_cross_experiment") as f:
        lines = f.readlines()
        for l in lines:
            m = reg.match(l)
            base, target, num = m.groups()
            db.add_transfer(base, target, Task.GRASPING.value, int(num) / 10.0)


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
            )
    populate_transfers(db)
    return db


if __name__ == "__main__":
    config = Config()
    db = create_and_populate_db(config)
    print(db.get_all_object_names_for_task(Task.GRASPING.value))
    print(db.get_all_object_names_for_task(Task.PUSHING.value))
    print(db.get_demo("banana", Task.GRASPING.value))
    print(db.get_demo("banana", Task.PUSHING.value))
