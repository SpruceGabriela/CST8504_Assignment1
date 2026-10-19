import pytest

from src.predictor import Predictor

# I already knew testing concepts, but pytest was new to me. The tests follow the steps from this article:
# https://medium.com/@bressan/software-engineering-for-data-scientists-testing-ml-models-with-pytest-d28b83ab54d9

# Fake day fro the test cases
MOCK_DAY = {
    "ghi": 5.5,
    "ghi_clear": 6.8,
    "temp": 28,
    "humidity": 65,
    "wind": 3.2,
    "clouds": 40,
    "rain": 1,
    "date": "2025-08-15",
}


def test_prediction_is_not_negative():
    predictor = Predictor()

    kwh = predictor.predict_from_inputs(**MOCK_DAY)
    assert isinstance(kwh, float)
    assert kwh >= 0


def test_ghi_zero_raises_error():
    predictor = Predictor()
    zero_clear_sky = {**MOCK_DAY, "ghi_clear": 0}

    with pytest.raises(ValueError):
        predictor.predict_from_inputs(**zero_clear_sky)


def test_missing_model_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        Predictor(model_path=tmp_path / "something.pkl")
