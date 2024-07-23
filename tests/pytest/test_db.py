import os

import pytest

from config import Config
from database import DB
from task_types import Task


@pytest.fixture
def db_fixture():
    config = Config()
    db = DB(config, "test.db")
    db.create_tables()
    yield db
    db.remove_all_tables()
    os.remove(os.path.join(config.BASE_DIR, "test.db"))


def test_add_object_works(db_fixture):
    db_fixture.add_object("object_name_1", "_test_assets/urdf1.urdf", "test")


@pytest.fixture
def db_fixture_objects(db_fixture):
    db_fixture.add_object("object_name_1", "_test_assets/urdf1.urdf", "test")
    db_fixture.add_object("object_name_2", "_test_assets/urdf2.urdf", "test")
    return db_fixture


def test_add_demo_works(db_fixture_objects):
    db_fixture_objects.add_demo(
        "object_name_1",
        Task.GRASPING.value,
        "tests/_test_assets/demo1.json",
        1.23,
        (1, 2, 3),
        (1.1, 2.2, 3.3),
    )


def test_add_transfer_works(db_fixture_objects):
    db_fixture_objects.add_transfer(
        "object_name_1", "object_name_2", Task.GRASPING.value, 0.123
    )


@pytest.fixture
def db_fixture_pop(db_fixture_objects):
    db_fixture_objects.add_demo(
        "object_name_1",
        Task.GRASPING.value,
        "tests/_test_assets/demo1.json",
        1.23,
        (1, 2, 3),
        (1.1, 2.2, 3.3),
        (1.11, 2.22, 3.33),
    )
    db_fixture_objects.add_demo(
        "object_name_2",
        Task.GRASPING.value,
        "tests/_test_assets/demo2.json",
        1.23,
        (1, 2, 3),
        (1.1, 2.2, 3.3),
        (1.11, 2.22, 3.33),
    )
    db_fixture_objects.add_demo(
        "object_name_1",
        Task.PUSHING.value,
        "tests/_test_assets/demo3.json",
    )
    db_fixture_objects.add_demo(
        "object_name_2",
        Task.PUSHING.value,
        "tests/_test_assets/demo4.json",
    )
    db_fixture_objects.add_transfer(
        "object_name_1", "object_name_2", Task.GRASPING.value, 0.123
    )
    db_fixture_objects.add_transfer(
        "object_name_1", "object_name_2", Task.PUSHING.value, 0.321
    )
    return db_fixture_objects


def test_get_demo_works(db_fixture_pop):
    assert (
        db_fixture_pop.get_demo("object_name_1", Task.GRASPING.value)
        == "tests/_test_assets/demo1.json"
    )
    assert (
        db_fixture_pop.get_demo("object_name_2", Task.GRASPING.value)
        == "tests/_test_assets/demo2.json"
    )
    assert (
        db_fixture_pop.get_demo("object_name_1", Task.PUSHING.value)
        == "tests/_test_assets/demo3.json"
    )
    assert (
        db_fixture_pop.get_demo("object_name_2", Task.PUSHING.value)
        == "tests/_test_assets/demo4.json"
    )


def test_get_load_info(db_fixture_pop):
    assert db_fixture_pop.get_load_info("object_name_1", Task.GRASPING.value) == (
        "tests/_test_assets/urdf1.urdf",
        1.23,
        (1, 2, 3),
        (1.1, 2.2, 3.3),
        (1.11, 2.22, 3.33),
    )
    assert db_fixture_pop.get_load_info("object_name_2", Task.GRASPING.value) == (
        "tests/_test_assets/urdf2.urdf",
        1.23,
        (1, 2, 3),
        (1.1, 2.2, 3.3),
        (1.11, 2.22, 3.33),
    )
    assert db_fixture_pop.get_load_info("object_name_1", Task.PUSHING.value) == (
        "tests/_test_assets/urdf1.urdf",
        1.0,
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
    )
    assert db_fixture_pop.get_load_info("object_name_2", Task.PUSHING.value) == (
        "tests/_test_assets/urdf2.urdf",
        1.0,
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
    )


def test_get_success_rate_works(db_fixture_pop):
    assert (
        db_fixture_pop.get_success_rate(
            "object_name_1", "object_name_2", Task.GRASPING.value
        )
        == 0.123
    )
    assert (
        db_fixture_pop.get_success_rate(
            "object_name_1", "object_name_2", Task.PUSHING.value
        )
        == 0.321
    )


def test_get_all_object_names_for_task_works(db_fixture_pop):
    assert db_fixture_pop.get_all_object_names_for_task(Task.PUSHING.value) == [
        "object_name_1",
        "object_name_2",
    ]


def test_add_object_with_same_name_updates(db_fixture_pop):
    assert db_fixture_pop.get_load_info("object_name_1", Task.GRASPING.value) == (
        "tests/_test_assets/urdf1.urdf",
        1.23,
        (1, 2, 3),
        (1.1, 2.2, 3.3),
        (1.11, 2.22, 3.33),
    )
    db_fixture_pop.add_object("object_name_1", "_test_assets/urdf3.urdf", "test")
    assert db_fixture_pop.get_load_info("object_name_1", Task.GRASPING.value) == (
        "tests/_test_assets/urdf3.urdf",
        1.23,
        (1, 2, 3),
        (1.1, 2.2, 3.3),
        (1.11, 2.22, 3.33),
    )


def test_add_demo_with_same_name_and_task_updates(db_fixture_pop):
    assert (
        db_fixture_pop.get_demo("object_name_1", Task.GRASPING.value)
        == "tests/_test_assets/demo1.json"
    )
    db_fixture_pop.add_demo(
        "object_name_1",
        Task.GRASPING.value,
        "tests/_test_assets/demo5.json",
        1.23,
        (1, 2, 3),
        (1.1, 2.2, 3.3),
        (1.11, 2.22, 3.33),
    )
    assert (
        db_fixture_pop.get_demo("object_name_1", Task.GRASPING.value)
        == "tests/_test_assets/demo5.json"
    )


def test_add_transfer_with_same_names_and_task_updates(db_fixture_pop):
    assert (
        db_fixture_pop.get_success_rate(
            "object_name_1", "object_name_2", Task.GRASPING.value
        )
        == 0.123
    )
    db_fixture_pop.add_transfer(
        "object_name_1", "object_name_2", Task.GRASPING.value, 0.124
    )
    assert (
        db_fixture_pop.get_success_rate(
            "object_name_1", "object_name_2", Task.GRASPING.value
        )
        == 0.124
    )
