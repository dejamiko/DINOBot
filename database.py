import os
import sqlite3

from config import Config


class DB:
    def __init__(self, config, name="dino.db"):
        self.config = config
        self.con = sqlite3.connect(name)

    def create_tables(self):
        try:
            self.con.execute(
                "CREATE TABLE demonstrations"
                "("
                "object_id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "object_name VARCHAR(50) UNIQUE NOT NULL, "
                "demo_path VARCHAR(100) UNIQUE NOT NULL "
                ")"
            )
        except sqlite3.OperationalError as e:
            if str(e) == "table demonstrations already exists":
                pass
            else:
                raise e
        try:
            self.con.execute(
                "CREATE TABLE transfers"
                "("
                "object_id_1 INT NOT NULL,"
                "object_id_2 INT NOT NULL,"
                "success_rate FLOAT NOT NULL,"
                "FOREIGN KEY (object_id_1) REFERENCES demonstrations(object_id),"
                "FOREIGN KEY (object_id_2) REFERENCES demonstrations(object_id)"
                ")")
        except sqlite3.OperationalError as e:
            if str(e) == "table transfers already exists":
                pass
            else:
                raise e

    def add_demo(self, object_name, demo_path):
        # self._check_paths(demo_path)
        # if there already exists a demo in the database with the given object name, override it
        if self.get_demo_for_object(object_name) is not None:
            with self.con:
                self.con.execute("UPDATE demonstrations SET demo_path=? WHERE object_name=?",
                                 (demo_path, object_name))
        else:
            with self.con:
                self.con.execute("INSERT INTO demonstrations VALUES(?, ?, ?)",
                                 (None, object_name, demo_path))

    def get_demo_for_object(self, object_name):
        # Note, those parameters are tuples with length one, otherwise the string is treated as a list of chars
        res = self.con.execute("SELECT demo_path FROM demonstrations WHERE object_name=?", (object_name,))
        res = res.fetchone()
        return res[0] if res is not None else None

    def _get_object_id_by_name(self, object_name):
        res = self.con.execute("SELECT object_id FROM demonstrations WHERE object_name=?", (object_name,))
        res = res.fetchone()
        return res[0] if res is not None else None

    def get_success_rate_for_objects(self, object_1_name, object_2_name):
        object_1_id = self._get_object_id_by_name(object_1_name)
        object_2_id = self._get_object_id_by_name(object_2_name)

        assert object_1_id is not None, f"There was no object named {object_1_name} in the database"
        assert object_2_id is not None, f"There was no object named {object_2_name} in the database"

        res = self.con.execute("SELECT success_rate FROM transfers WHERE (object_id_1=? AND object_id_2=?)",
                               (object_1_id, object_2_id))
        res = res.fetchone()
        return res[0] if res is not None else None

    def add_transfer(self, object_1_name, object_2_name, success_rate):
        assert 0 <= success_rate <= 1.0, "The success rate should be between 0 and 1"
        object_1_id = self._get_object_id_by_name(object_1_name)
        object_2_id = self._get_object_id_by_name(object_2_name)

        assert object_1_id is not None, f"There was no object named {object_1_name} in the database"
        assert object_2_id is not None, f"There was no object named {object_2_name} in the database"

        prev = self.get_success_rate_for_objects(object_1_name, object_2_name)

        if prev is not None:
            with self.con:
                self.con.execute("UPDATE transfers SET success_rate=? WHERE object_id_1=? AND object_id_2=?",
                                 (success_rate, object_1_id, object_2_id))
        else:
            with self.con:
                self.con.execute("INSERT INTO transfers VALUES(?, ?, ?)", (object_1_id, object_2_id, success_rate))

    @staticmethod
    def _check_paths(demo_path):
        assert os.path.isfile(demo_path)
        assert demo_path.endswith(".json")

    def remove_all_tables(self):
        with self.con:
            try:
                self.con.execute("DROP TABLE demonstrations")
            except sqlite3.OperationalError as e:
                if str(e).startswith("no such table: "):
                    pass
                else:
                    raise e
            try:
                self.con.execute("DROP TABLE transfers")
            except sqlite3.OperationalError as e:
                if str(e).startswith("no such table: "):
                    pass
                else:
                    raise e


if __name__ == "__main__":
    config = Config()
    db = DB(config)

    db.remove_all_tables()
    db.create_tables()

    base_dir = "demonstrations/"

    for f in os.listdir(base_dir):
        if not f.endswith(".json"):
            continue
        object_name = "_".join(f.split("_")[1:])[:-5]
        db.add_demo(object_name, base_dir + f)

    # TODO add table for urdf info (object_name, urdf_path, scale)


