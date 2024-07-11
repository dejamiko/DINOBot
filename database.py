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
                "demo_path VARCHAR(100) UNIQUE NOT NULL, "
                "image_dir_path VARCHAR(100) UNIQUE NOT NULL"
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

    def add_demo(self, object_name, demo_path, image_dir_path):
        # self._check_paths(demo_path, image_dir_path)
        # if there already exists a demo in the database with the given object name, override it
        if self.get_demo_for_object(object_name) is not None:
            with self.con:
                self.con.execute("UPDATE demonstrations SET demo_path=?, image_dir_path=? WHERE object_name=?",
                                 (demo_path, image_dir_path, object_name))
        else:
            with self.con:
                self.con.execute("INSERT INTO demonstrations VALUES(?, ?, ?, ?)",
                                 (None, object_name, demo_path, image_dir_path))

    def get_demo_for_object(self, object_name):
        # Note, those parameters are tuples with length one, otherwise the string is treated as a list of chars
        res = self.con.execute("SELECT demo_path FROM demonstrations WHERE object_name=?", (object_name,))
        res = res.fetchone()
        return res[0] if res is not None else None

    def get_image_dir_path_for_object(self, object_name):
        res = self.con.execute("SELECT image_dir_path FROM demonstrations WHERE object_name=?", (object_name,))
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

    def _check_paths(self, demo_path, images_dir_path):
        assert os.path.isfile(demo_path)
        assert demo_path.endswith(".json")

        assert os.path.isdir(images_dir_path)
        assert len(os.listdir(images_dir_path)) == len(self.config.IMAGE_VIEWPOINTS)

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

    db.add_demo("object_name1", "demo_path1", "images_path1")
    db.add_demo("object_name2", "demo_path2", "images_path2")
    db.add_demo("object_name3", "demo_path3", "images_path3")
    try:
        db.add_demo("object_name1", "demo_path3", "images_path3")
    except sqlite3.IntegrityError as e:
        if str(e).startswith("UNIQUE constraint failed: "):
            pass
        else:
            raise e

    print(db.get_demo_for_object("object_name1"))
    db.add_demo("object_name1", "demo_path4", "images_path4")
    print(db.get_demo_for_object("object_name1"))

    db.add_transfer("object_name1", "object_name2", 0.7)
    db.add_transfer("object_name2", "object_name3", 0.9)
    db.add_transfer("object_name2", "object_name3", 0.9)
    print(db.get_success_rate_for_objects("object_name1", "object_name2"))
    db.add_transfer("object_name1", "object_name2", 0.1)
    print(db.get_success_rate_for_objects("object_name1", "object_name2"))
