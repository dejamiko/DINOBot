import os
import sqlite3

import pytest

from config import Config
from database import DB


@pytest.fixture
def db_fixture():
    config = Config()
    db = DB(config, "test.db")
    db.create_tables()
    yield db
    db.remove_all_tables()
    os.remove("test.db")


def test_can_add_demos(db_fixture):
    db_fixture.add_demo("object_name1", "demo_path1", "images_path1")
    db_fixture.add_demo("object_name2", "demo_path2", "images_path2")


def test_object_demo_path_is_unique(db_fixture):
    db_fixture.add_demo("object_name1", "demo_path1", "images_path1")
    with pytest.raises(sqlite3.IntegrityError) as e:
        db_fixture.add_demo("object_name2", "demo_path1", "images_path2")
    assert str(e.value) == "UNIQUE constraint failed: demonstrations.demo_path"


def test_object_image_dir_path_is_unique(db_fixture):
    db_fixture.add_demo("object_name1", "demo_path1", "images_path1")
    with pytest.raises(sqlite3.IntegrityError) as e:
        db_fixture.add_demo("object_name2", "demo_path2", "images_path1")
    assert str(e.value) == "UNIQUE constraint failed: demonstrations.image_dir_path"


def test_get_demo_works(db_fixture):
    db_fixture.add_demo("object_name1", "demo_path1", "images_path1")
    assert db_fixture.get_demo_for_object("object_name1") == "demo_path1"


def test_adding_with_existing_object_name_updates(db_fixture):
    db_fixture.add_demo("object_name1", "demo_path1", "images_path1")
    assert db_fixture.get_demo_for_object("object_name1") == "demo_path1"
    db_fixture.add_demo("object_name1", "demo_path2", "images_path1")
    assert db_fixture.get_demo_for_object("object_name1") == "demo_path2"


def test_get_image_dir_works(db_fixture):
    db_fixture.add_demo("object_name1", "demo_path1", "images_path1")
    assert db_fixture.get_image_dir_path_for_object("object_name1") == "images_path1"


def test_add_transfer_works(db_fixture):
    db_fixture.add_demo("object_name1", "demo_path1", "images_path1")
    db_fixture.add_demo("object_name2", "demo_path2", "images_path2")
    db_fixture.add_transfer("object_name1", "object_name2", 0.1)


def test_get_success_rate_works(db_fixture):
    db_fixture.add_demo("object_name1", "demo_path1", "images_path1")
    db_fixture.add_demo("object_name2", "demo_path2", "images_path2")
    db_fixture.add_transfer("object_name1", "object_name2", 0.1)
    assert db_fixture.get_success_rate_for_objects("object_name1", "object_name2") == 0.1


def test_add_transfer_for_existing_objects_updates(db_fixture):
    db_fixture.add_demo("object_name1", "demo_path1", "images_path1")
    db_fixture.add_demo("object_name2", "demo_path2", "images_path2")
    db_fixture.add_transfer("object_name1", "object_name2", 0.1)
    assert db_fixture.get_success_rate_for_objects("object_name1", "object_name2") == 0.1
    db_fixture.add_transfer("object_name1", "object_name2", 0.7)
    assert db_fixture.get_success_rate_for_objects("object_name1", "object_name2") == 0.7
