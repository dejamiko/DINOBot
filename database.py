import os
import re
import sqlite3

import pybullet_data
from pybullet_object_models import ycb_objects

from config import Config


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
            "CREATE TABLE demonstrations"
            "("
            "object_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "object_name VARCHAR(50) UNIQUE NOT NULL, "
            "demo_path VARCHAR(100) UNIQUE NOT NULL "
            ")"
        )
        self._create_table("demonstrations", sql_string)

        sql_string = (
            "CREATE TABLE transfers"
            "("
            "object_id_1 INT NOT NULL,"
            "object_id_2 INT NOT NULL,"
            "success_rate FLOAT NOT NULL,"
            "FOREIGN KEY (object_id_1) REFERENCES demonstrations(object_id),"
            "FOREIGN KEY (object_id_2) REFERENCES demonstrations(object_id)"
            ")"
        )
        self._create_table("transfers", sql_string)

        sql_string = (
            "CREATE TABLE urdf_info"
            "("
            "object_id INT NOT NULL,"
            "urdf_path VARCHAR(100) NOT NULL,"
            "scale FLOAT NOT NULL,"
            "source VARCHAR(20) NOT NULL,"
            "FOREIGN KEY (object_id) REFERENCES demonstrations(object_id)"
            ")"
        )
        self._create_table("urdf_info", sql_string)

    def _create_table(self, table_name, sql_string):
        try:
            self.con.execute(sql_string)
        except sqlite3.OperationalError as e:
            if str(e) == f"table {table_name} already exists":
                pass
            else:
                raise e

    def add_demo(self, object_name, demo_path):
        self._check_paths(demo_path, ".json")
        # if there already exists a demo in the database with the given object name, override it
        if self.get_demo_for_object(object_name) is not None:
            with self.con as c:
                c.execute(
                    "UPDATE demonstrations SET demo_path=? WHERE object_name=?",
                    (demo_path, object_name),
                )
        else:
            with self.con as c:
                c.execute(
                    "INSERT INTO demonstrations VALUES(?, ?, ?)",
                    (None, object_name, demo_path),
                )

    def get_demo_for_object(self, object_name):
        # Note, those parameters are tuples with length one, otherwise the string is treated as a list of chars
        res = self.con.execute(
            "SELECT demo_path FROM demonstrations WHERE object_name=?", (object_name,)
        )
        res = res.fetchone()
        return res[0] if res is not None else None

    def _get_object_id_by_name(self, object_name):
        res = self.con.execute(
            "SELECT object_id FROM demonstrations WHERE object_name=?", (object_name,)
        )
        res = res.fetchone()
        return res[0] if res is not None else None

    def get_success_rate_for_objects(self, object_1_name, object_2_name):
        object_1_id = self._get_object_id_by_name(object_1_name)
        object_2_id = self._get_object_id_by_name(object_2_name)

        assert (
            object_1_id is not None
        ), f"There was no object named {object_1_name} in the database"
        assert (
            object_2_id is not None
        ), f"There was no object named {object_2_name} in the database"

        res = self.con.execute(
            "SELECT success_rate FROM transfers WHERE (object_id_1=? AND object_id_2=?)",
            (object_1_id, object_2_id),
        )
        res = res.fetchone()
        return res[0] if res is not None else None

    def add_transfer(self, object_1_name, object_2_name, success_rate):
        assert 0 <= success_rate <= 1.0, "The success rate should be between 0 and 1"
        object_1_id = self._get_object_id_by_name(object_1_name)
        object_2_id = self._get_object_id_by_name(object_2_name)

        assert (
            object_1_id is not None
        ), f"There was no object named {object_1_name} in the database"
        assert (
            object_2_id is not None
        ), f"There was no object named {object_2_name} in the database"

        prev = self.get_success_rate_for_objects(object_1_name, object_2_name)

        if prev is not None:
            with self.con as c:
                c.execute(
                    "UPDATE transfers SET success_rate=? WHERE object_id_1=? AND object_id_2=?",
                    (success_rate, object_1_id, object_2_id),
                )
        else:
            with self.con as c:
                c.execute(
                    "INSERT INTO transfers VALUES(?, ?, ?)",
                    (object_1_id, object_2_id, success_rate),
                )

    @staticmethod
    def _check_paths(demo_path, extension):
        assert os.path.isfile(demo_path)
        assert demo_path.endswith(extension)

    def remove_all_tables(self):
        with self.con:
            for table in ["demonstrations", "transfers", "urdf_info"]:
                try:
                    self.con.execute(f"DROP TABLE {table}")
                except sqlite3.OperationalError as e:
                    if str(e).startswith("no such table: "):
                        pass
                    else:
                        raise e

    def add_urdf_info(self, object_name, urdf_path, scale, source):
        self._check_paths(os.path.join(get_path(source), urdf_path), ".urdf")

        rel_urdf_path = os.path.relpath(urdf_path, "/Users/mikolajdeja/Coding/DINOBot")

        object_id = self._get_object_id_by_name(object_name)

        assert (
            object_id is not None
        ), f"There was no object named {object_name} in the database"

        prev = self.get_urdf_path(object_name)

        if prev is not None:
            with self.con as c:
                c.execute(
                    "UPDATE urdf_info SET urdf_path=?, scale=?, source=? WHERE object_id=?",
                    (rel_urdf_path, scale, source, object_id),
                )
        else:
            with self.con as c:
                c.execute(
                    "INSERT INTO urdf_info VALUES(?, ?, ?, ?)",
                    (object_id, rel_urdf_path, scale, source),
                )

    def get_urdf_path(self, object_name):
        object_id = self._get_object_id_by_name(object_name)

        assert (
            object_id is not None
        ), f"There was no object named {object_id} in the database"

        res = self.con.execute(
            "SELECT source, urdf_path FROM urdf_info WHERE object_id=?", (object_id,)
        )
        res = res.fetchone()
        return os.path.join(get_path(res[0]), res[1]) if res is not None else None

    def get_urdf_scale(self, object_name):
        object_id = self._get_object_id_by_name(object_name)

        assert (
            object_id is not None
        ), f"There was no object named {object_id} in the database"

        res = self.con.execute(
            "SELECT scale FROM urdf_info WHERE object_id=?", (object_id,)
        )
        res = res.fetchone()
        return res[0] if res is not None else None

    def get_all_object_names(self):
        res = self.con.execute(
            "SELECT object_name "
            "FROM demonstrations "
            "WHERE NOT object_name LIKE '%\_OLD%' ESCAPE '\\' "
            "ORDER BY object_name"
        )
        return [r[0] for r in res.fetchall()]


def populate_urdf_info(db):
    # YCB objects
    names = [
        "YcbBanana",
        "YcbChipsCan",
        "YcbCrackerBox",
        "YcbFoamBrick",
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

    scales = {"YcbChipsCan": 0.7}

    for n in names:
        insert_ycb_object(db, n, scales.get(n, 1.0))

    # pybullet data objects
    db.add_urdf_info("bike", "bicycle/bike.urdf", 0.1, "pybullet_data")
    db.add_urdf_info("domino", "domino/domino.urdf", 2.5, "pybullet_data")
    db.add_urdf_info("duck", "duck_vhacd.urdf", 1.0, "pybullet_data")
    db.add_urdf_info("jenga", "jenga/jenga.urdf", 1.0, "pybullet_data")
    db.add_urdf_info(
        "mini_cheetah", "mini_cheetah/mini_cheetah.urdf", 0.25, "pybullet_data"
    )
    db.add_urdf_info("minitaur", "quadruped/minitaur.urdf", 0.25, "pybullet_data")
    db.add_urdf_info("mug", "urdf/mug.urdf", 1.0, "pybullet_data")
    db.add_urdf_info("racecar", "racecar/racecar.urdf", 0.2, "pybullet_data")


def insert_ycb_object(db, obj_name, scale=1.0):
    path_to_urdf = os.path.join(obj_name, "model.urdf")
    snake_case_name = re.sub(r"(?<!^)(?=[A-Z])", "_", obj_name).lower()[4:]
    db.add_urdf_info(snake_case_name, path_to_urdf, scale, "ycb")


def create_and_populate_db(config):
    db = DB(config)
    db.remove_all_tables()
    db.create_tables()
    base_dir = "demonstrations/"
    for f in os.listdir(base_dir):
        if not f.endswith(".json"):
            continue
        object_name = "_".join(f.split("_")[1:])[:-5]
        db.add_demo(object_name, base_dir + f)
    populate_urdf_info(db)
    return db


if __name__ == "__main__":
    config = Config()
    db = create_and_populate_db(config)
    print(db.get_all_object_names())
