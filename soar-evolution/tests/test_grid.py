"""Tests for Grid, GridDiff, and ARCTask dataclasses."""

import pytest
from src.arc.grid import Grid, GridDiff, ARCExample, ARCTask, diff_grids


class TestGrid:
    def test_create_grid(self):
        g = Grid.from_list([[1, 2], [3, 4]])
        assert g.height == 2
        assert g.width == 2
        assert g.shape == (2, 2)

    def test_empty_grid_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            Grid(data=[])

    def test_jagged_grid_raises(self):
        with pytest.raises(ValueError, match="Row 1"):
            Grid(data=[[1, 2], [3]])

    def test_to_list(self):
        data = [[1, 2], [3, 4]]
        g = Grid.from_list(data)
        result = g.to_list()
        assert result == data
        # Ensure deep copy
        result[0][0] = 999
        assert g.data[0][0] == 1

    def test_to_ascii(self):
        g = Grid.from_list([[1, 0], [0, 2]])
        ascii_repr = g.to_ascii()
        assert ascii_repr == "10\n02"

    def test_equality(self):
        g1 = Grid.from_list([[1, 2], [3, 4]])
        g2 = Grid.from_list([[1, 2], [3, 4]])
        g3 = Grid.from_list([[1, 2], [3, 5]])
        assert g1 == g2
        assert g1 != g3

    def test_equality_not_grid(self):
        g = Grid.from_list([[1]])
        assert g.__eq__("not a grid") is NotImplemented

    def test_hash(self):
        g1 = Grid.from_list([[1, 2], [3, 4]])
        g2 = Grid.from_list([[1, 2], [3, 4]])
        assert hash(g1) == hash(g2)

    def test_get_set(self):
        g = Grid.from_list([[1, 2], [3, 4]])
        assert g.get(0, 1) == 2
        g.set(0, 1, 9)
        assert g.get(0, 1) == 9

    def test_copy(self):
        g = Grid.from_list([[1, 2], [3, 4]])
        g2 = g.copy()
        assert g == g2
        g2.set(0, 0, 999)
        assert g != g2

    def test_colors_used(self):
        g = Grid.from_list([[0, 1, 2], [3, 0, 1]])
        assert g.colors_used() == {0, 1, 2, 3}

    def test_pixel_count(self):
        g = Grid.from_list([[1, 2, 3], [4, 5, 6]])
        assert g.pixel_count() == 6

    def test_zeros(self):
        g = Grid.zeros(3, 4)
        assert g.shape == (3, 4)
        assert all(cell == 0 for row in g.data for cell in row)


class TestGridDiff:
    def test_same_grids(self):
        g = Grid.from_list([[1, 2], [3, 4]])
        d = diff_grids(g, g)
        assert d.num_changes == 0
        assert d.change_ratio == 0.0
        assert not d.shape_changed

    def test_different_values(self):
        g1 = Grid.from_list([[1, 2], [3, 4]])
        g2 = Grid.from_list([[1, 9], [3, 4]])
        d = diff_grids(g1, g2)
        assert d.num_changes == 1
        assert not d.shape_changed
        assert d.changed_cells[0] == (0, 1, 2, 9)

    def test_different_shapes(self):
        g1 = Grid.from_list([[1, 2], [3, 4]])
        g2 = Grid.from_list([[1, 2, 5], [3, 4, 6]])
        d = diff_grids(g1, g2)
        assert d.shape_changed
        assert d.old_shape == (2, 2)
        assert d.new_shape == (2, 3)

    def test_summary(self):
        g1 = Grid.from_list([[1, 2], [3, 4]])
        g2 = Grid.from_list([[1, 9], [3, 4]])
        d = diff_grids(g1, g2)
        s = d.summary()
        assert "Changed cells: 1" in s

    def test_shape_changed_summary(self):
        g1 = Grid.from_list([[1, 2], [3, 4]])
        g2 = Grid.from_list([[1], [3]])
        d = diff_grids(g1, g2)
        s = d.summary()
        assert "Shape:" in s

    def test_larger_grid_a(self):
        g1 = Grid.from_list([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        g2 = Grid.from_list([[1, 2], [4, 5]])
        d = diff_grids(g1, g2)
        assert d.shape_changed
        assert d.num_changes > 0

    def test_larger_grid_b(self):
        g1 = Grid.from_list([[1, 2], [4, 5]])
        g2 = Grid.from_list([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        d = diff_grids(g1, g2)
        assert d.shape_changed


class TestARCTask:
    def test_from_dict(self):
        data = {
            "train": [
                {"input": [[1, 0], [0, 1]], "output": [[2, 0], [0, 2]]},
            ],
            "test": [
                {"input": [[0, 1], [1, 0]], "output": [[0, 2], [2, 0]]},
            ],
        }
        task = ARCTask.from_dict("test_task", data)
        assert task.task_id == "test_task"
        assert task.num_train == 1
        assert task.num_test == 1

    def test_to_dict(self):
        data = {
            "train": [
                {"input": [[1, 0]], "output": [[2, 0]]},
            ],
            "test": [],
        }
        task = ARCTask.from_dict("t", data)
        result = task.to_dict()
        assert result["train"][0]["input"] == [[1, 0]]
        assert result["train"][0]["output"] == [[2, 0]]
        assert result["test"] == []

    def test_all_examples(self):
        data = {
            "train": [
                {"input": [[1]], "output": [[2]]},
                {"input": [[3]], "output": [[4]]},
            ],
            "test": [
                {"input": [[5]], "output": [[6]]},
            ],
        }
        task = ARCTask.from_dict("t", data)
        assert len(task.all_examples()) == 3

    def test_metadata(self):
        data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [],
            "metadata": {"source": "test"},
        }
        task = ARCTask.from_dict("t", data)
        assert task.metadata == {"source": "test"}

    def test_from_dict_empty_metadata(self):
        data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [],
        }
        task = ARCTask.from_dict("t", data)
        assert task.metadata == {}
