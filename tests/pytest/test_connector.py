import pytest

import connector
from config import Config


@pytest.fixture
def connector_fixture():
    config = Config()
    config.LOAD_SIZE = 40
    config.RUN_LOCALLY = True
    config.DEVICE = "cpu"
    image1 = "tests/assets/image1.png"
    image2 = "tests/assets/image2.png"

    # TODO mock the server somehow and assert the results are the same

    yield config, image1, image2


def test_get_correspondences_slow_no_draw_returns_correct_results(connector_fixture):
    config, image1, image2 = connector_fixture
    results = connector.get_correspondences(config, 0, image1, image2)
    assert len(results) == 3
    assert "points1_2d" in results
    assert "points2_2d" in results
    assert "time_taken" in results


def test_get_correspondences_slow_draw_returns_correct_results(connector_fixture):
    config, image1, image2 = connector_fixture
    config.DRAW_CORRESPONDENCES = True
    results = connector.get_correspondences(config, 0, image1, image2)
    assert len(results) == 5
    assert "points1_2d" in results
    assert "points2_2d" in results
    assert "time_taken" in results
    assert "image1_correspondences" in results
    assert "image2_correspondences" in results


def test_get_correspondences_fast_no_draw_returns_correct_results(connector_fixture):
    config, image1, image2 = connector_fixture
    config.USE_FAST_CORRESPONDENCES = True
    results = connector.get_correspondences(config, 0, image1, image2)
    assert len(results) == 5
    assert "points1_2d" in results
    assert "points2_2d" in results
    assert "time_taken" in results
    assert "num_patches" in results
    assert "descriptor_vectors" in results


def test_get_correspondences_fast_draw_returns_correct_results(connector_fixture):
    config, image1, image2 = connector_fixture
    config.USE_FAST_CORRESPONDENCES = True
    config.DRAW_CORRESPONDENCES = True
    results = connector.get_correspondences(config, 0, image1, image2)
    assert len(results) == 7
    assert "points1_2d" in results
    assert "points2_2d" in results
    assert "time_taken" in results
    assert "num_patches" in results
    assert "descriptor_vectors" in results
    assert "image1_correspondences" in results
    assert "image2_correspondences" in results


def test_get_correspondences_fast_no_draw_no_recalc_returns_correct_results(
    connector_fixture,
):
    config, image1, image2 = connector_fixture
    config.USE_FAST_CORRESPONDENCES = True
    # run the previous one
    results = connector.get_correspondences(config, 0, image1, image2)
    # run the actual test one
    results = connector.get_correspondences(
        config,
        2,
        image1,
        image2,
        points1_2d=results["points1_2d"],
        num_patches=results["num_patches"],
        descriptor_vectors=results["descriptor_vectors"],
    )
    assert "points1_2d" in results
    assert "points2_2d" in results
    assert "time_taken" in results
    assert "num_patches" in results
    assert "descriptor_vectors" in results
    assert len(results) == 5
