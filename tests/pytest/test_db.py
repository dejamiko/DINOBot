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
    db_fixture.add_demo("object_name1", "tests/assets/demo1.json")
    db_fixture.add_demo("object_name2", "tests/assets/demo2.json")
    yield db_fixture


def test_can_add_demos(db_fixture):
    db_fixture.add_demo("object_name1", "tests/assets/demo1.json")
    db_fixture.add_demo("object_name2", "tests/assets/demo2.json")


def test_object_demo_path_is_unique(db_with_demos_fixture):
    with pytest.raises(sqlite3.IntegrityError) as e:
        db_with_demos_fixture.add_demo("object_name3", "tests/assets/demo1.json")
    assert str(e.value) == "UNIQUE constraint failed: demonstrations.demo_path"


def test_get_demo_works(db_with_demos_fixture):
    assert (
        db_with_demos_fixture.get_demo_for_object("object_name1")
        == "tests/assets/demo1.json"
    )


def test_adding_with_existing_object_name_updates(db_with_demos_fixture):
    assert (
        db_with_demos_fixture.get_demo_for_object("object_name1")
        == "tests/assets/demo1.json"
    )
    db_with_demos_fixture.add_demo("object_name1", "tests/assets/demo3.json")
    assert (
        db_with_demos_fixture.get_demo_for_object("object_name1")
        == "tests/assets/demo3.json"
    )


def test_add_transfer_works(db_with_demos_fixture):
    db_with_demos_fixture.add_transfer("object_name1", "object_name2", 0.1)


def test_get_success_rate_works(db_with_demos_fixture):
    db_with_demos_fixture.add_transfer("object_name1", "object_name2", 0.1)
    assert (
        db_with_demos_fixture.get_success_rate_for_objects(
            "object_name1", "object_name2"
        )
        == 0.1
    )


def test_get_success_rate_for_wrong_name_fails(db_with_demos_fixture):
    with pytest.raises(AssertionError):
        db_with_demos_fixture.get_success_rate_for_objects(
            "object_name3", "object_name1"
        )


def test_add_transfer_for_existing_objects_updates(db_with_demos_fixture):
    db_with_demos_fixture.add_transfer("object_name1", "object_name2", 0.1)
    assert (
        db_with_demos_fixture.get_success_rate_for_objects(
            "object_name1", "object_name2"
        )
        == 0.1
    )
    db_with_demos_fixture.add_transfer("object_name1", "object_name2", 0.7)
    assert (
        db_with_demos_fixture.get_success_rate_for_objects(
            "object_name1", "object_name2"
        )
        == 0.7
    )


def test_add_urdf_info_works(db_with_demos_fixture):
    db_with_demos_fixture.add_urdf_info("object_name1", "tests/assets/urdf1.urdf", 1.0)
    db_with_demos_fixture.add_urdf_info("object_name2", "tests/assets/urdf2.urdf", 1.0)


def test_add_urdf_info_with_wrong_path_fails(db_with_demos_fixture):
    with pytest.raises(AssertionError):
        db_with_demos_fixture.add_urdf_info(
            "object_name1", "tests/assets/urdf3.urdf", 1.0
        )


def test_add_urdf_info_with_wrong_extension_fails(db_with_demos_fixture):
    with pytest.raises(AssertionError):
        db_with_demos_fixture.add_urdf_info(
            "object_name1", "tests/assets/demo1.json", 1.0
        )


def test_get_urdf_path_works(db_with_demos_fixture):
    db_with_demos_fixture.add_urdf_info("object_name1", "tests/assets/urdf1.urdf", 1.0)
    assert (
        db_with_demos_fixture.get_urdf_path("object_name1") == "tests/assets/urdf1.urdf"
    )


def test_get_urdf_scale_works(db_with_demos_fixture):
    db_with_demos_fixture.add_urdf_info("object_name1", "tests/assets/urdf1.urdf", 1.0)
    assert db_with_demos_fixture.get_urdf_scale("object_name1") == 1.0


def test_get_urdf_path_for_wrong_name_fails(db_with_demos_fixture):
    with pytest.raises(AssertionError):
        db_with_demos_fixture.get_urdf_path("object_name3")


def test_get_urdf_scale_for_wrong_name_fails(db_with_demos_fixture):
    with pytest.raises(AssertionError):
        db_with_demos_fixture.get_urdf_scale("object_name3")


def test_add_urdf_info_for_existing_object_updates(db_with_demos_fixture):
    db_with_demos_fixture.add_urdf_info("object_name1", "tests/assets/urdf1.urdf", 1.0)
    assert (
        db_with_demos_fixture.get_urdf_path("object_name1") == "tests/assets/urdf1.urdf"
    )
    assert db_with_demos_fixture.get_urdf_scale("object_name1") == 1.0
    db_with_demos_fixture.add_urdf_info("object_name1", "tests/assets/urdf2.urdf", 2.0)
    assert (
        db_with_demos_fixture.get_urdf_path("object_name1") == "tests/assets/urdf2.urdf"
    )
    assert db_with_demos_fixture.get_urdf_scale("object_name1") == 2.0
