"""Tests for the REPLSerializer."""

import pytest
from src.memory.serializer import REPLSerializer
from src.interface.errors import SerializationError


class TestREPLSerializer:
    """Test serialization round-trips."""

    def setup_method(self):
        self.serializer = REPLSerializer()

    def test_int_round_trip(self):
        value = 42
        result = self.serializer.round_trip(value)
        assert result == value

    def test_float_round_trip(self):
        value = 3.14
        result = self.serializer.round_trip(value)
        assert abs(result - value) < 1e-10

    def test_string_round_trip(self):
        value = "hello world"
        result = self.serializer.round_trip(value)
        assert result == value

    def test_bool_round_trip(self):
        assert self.serializer.round_trip(True) is True
        assert self.serializer.round_trip(False) is False

    def test_none_round_trip(self):
        assert self.serializer.round_trip(None) is None

    def test_list_round_trip(self):
        value = [1, 2, 3, "four", 5.0]
        result = self.serializer.round_trip(value)
        assert result == value

    def test_dict_round_trip(self):
        value = {"a": 1, "b": "two", "c": [3, 4]}
        result = self.serializer.round_trip(value)
        assert result == value

    def test_nested_structures(self):
        value = {"list": [1, 2, {"nested": True}], "num": 42}
        result = self.serializer.round_trip(value)
        assert result == value

    def test_numpy_array_round_trip(self):
        pytest.importorskip("numpy")
        import numpy as np
        value = np.array([1.0, 2.0, 3.0])
        result = self.serializer.round_trip(value)
        assert np.array_equal(result, value)

    def test_numpy_2d_array(self):
        pytest.importorskip("numpy")
        import numpy as np
        value = np.array([[1, 2], [3, 4]])
        result = self.serializer.round_trip(value)
        assert np.array_equal(result, value)

    def test_serialize_tag_json(self):
        tag, data = self.serializer.serialize(42)
        assert tag == "json"

    def test_serialize_tag_numpy(self):
        pytest.importorskip("numpy")
        import numpy as np
        tag, data = self.serializer.serialize(np.array([1, 2, 3]))
        assert tag == "numpy"

    def test_serialize_tag_dill(self):
        # Functions require dill
        def my_func():
            return 42
        tag, data = self.serializer.serialize(my_func)
        assert tag == "dill"

    def test_can_serialize(self):
        assert self.serializer.can_serialize(42)
        assert self.serializer.can_serialize("hello")
        assert self.serializer.can_serialize([1, 2, 3])

    def test_deserialize_unknown_tag(self):
        with pytest.raises(SerializationError):
            self.serializer.deserialize("unknown_tag", b"data")

    def test_deserialize_bad_data(self):
        with pytest.raises(SerializationError):
            self.serializer.deserialize("json", b"not json{{{")

    def test_empty_list(self):
        assert self.serializer.round_trip([]) == []

    def test_empty_dict(self):
        assert self.serializer.round_trip({}) == {}
