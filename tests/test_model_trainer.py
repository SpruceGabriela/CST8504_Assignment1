import pandas as pd
import pytest

from src.model_trainer import ModelTrainer

# Fake dataset to use in the test cases
MOCK_DAYS = pd.DataFrame(
    {
        "id": [1, 2, 3, 4, 5, 6, 7, 8],
        "date": [
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-04",
            "2025-01-05",
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
        ],
        "ghi": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 3.5],
        "temp": [25, 26, 27, 28, 29, 25, 26, 30],
        "label": [14.5, 17.0, 19.5, 22.0, 24.5, 24.5, 27.0, 22.0],
    }
)


def test_save_before_training_error(tmp_path):
    trainer = ModelTrainer(models_dir=tmp_path)
    with pytest.raises(RuntimeError):
        trainer.save_model()


def test_trainer_validation_and_test(tmp_path):
    MOCK_DAYS.iloc[0:4].to_csv(tmp_path / "train.csv", index=False)
    MOCK_DAYS.iloc[4:6].to_csv(tmp_path / "validation.csv", index=False)
    MOCK_DAYS.iloc[6:8].to_csv(tmp_path / "test.csv", index=False)
    trainer = ModelTrainer(data_dir=tmp_path, models_dir=tmp_path)

    trainer.load_data().train_model().evaluate_train_and_validation().evaluate_test().save_model()

    metrics_shape = {"MAE", "RMSE", "R2"}

    assert set(trainer.metrics["train"]) == metrics_shape
    assert set(trainer.metrics["validation"]) == metrics_shape
    assert set(trainer.metrics["test"]) == metrics_shape

    assert (tmp_path / "linear_regression.pkl").exists()
