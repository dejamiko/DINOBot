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


@pytest.fixture
def db_with_demos_fixture(db_fixture):
    db_fixture.add_demo("object_name1", "demo_path1")
    db_fixture.add_demo("object_name2", "demo_path2")
    yield db_fixture


def test_can_add_demos(db_fixture):
    db_fixture.add_demo("object_name1", "demo_path1")
    db_fixture.add_demo("object_name2", "demo_path2")


def test_object_demo_path_is_unique(db_with_demos_fixture):
    with pytest.raises(sqlite3.IntegrityError) as e:
        db_with_demos_fixture.add_demo("object_name3", "demo_path1")
    assert str(e.value) == "UNIQUE constraint failed: demonstrations.demo_path"


def test_get_demo_works(db_with_demos_fixture):
    assert db_with_demos_fixture.get_demo_for_object("object_name1") == "demo_path1"


def test_adding_with_existing_object_name_updates(db_with_demos_fixture):
    assert db_with_demos_fixture.get_demo_for_object("object_name1") == "demo_path1"
    db_with_demos_fixture.add_demo("object_name1", "demo_path3")
    assert db_with_demos_fixture.get_demo_for_object("object_name1") == "demo_path3"


def test_add_transfer_works(db_with_demos_fixture):
    db_with_demos_fixture.add_transfer("object_name1", "object_name2", 0.1)


def test_get_success_rate_works(db_with_demos_fixture):
    db_with_demos_fixture.add_transfer("object_name1", "object_name2", 0.1)
    assert db_with_demos_fixture.get_success_rate_for_objects("object_name1", "object_name2") == 0.1


def test_add_transfer_for_existing_objects_updates(db_with_demos_fixture):
    db_with_demos_fixture.add_transfer("object_name1", "object_name2", 0.1)
    assert db_with_demos_fixture.get_success_rate_for_objects("object_name1", "object_name2") == 0.1
    db_with_demos_fixture.add_transfer("object_name1", "object_name2", 0.7)
    assert db_with_demos_fixture.get_success_rate_for_objects("object_name1", "object_name2") == 0.7
