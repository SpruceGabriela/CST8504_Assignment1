# CST8504 Assignment 1 Prediction Dashboard

End-to-end machine learning project for CST8504 Assignment 1. It predicts the **daily energy generation (kWh)** of a solar plant in Salem, Tamil Nadu (India), from daily weather data. It includes a data cleaning pipeline, a Linear Regression model evaluated on a held-out test set, and a Streamlit dashboard for exploring the data and making live predictions, all packaged with Docker.

Repository: https://github.com/SpruceGabriela/CST8504_Assignment1

## 📊 Dataset
- **Weather (features):** NASA POWER API, daily data (Jul/2025 to Jun/2026): irradiance, temperature, humidity, wind, clouds, rain.
- **Target:** daily cleaned energy (kWh) from the Kaggle daily generation dataset.

---

## 🛠️ Tools & Technologies
- Python 3.10, Conda, Pip
- pandas, scikit-learn, Streamlit, Plotly
- Docker
- Ruff (linter & formatter), pytest

---

## 🚀 Getting Started
Clone the repository:
   ```bash
   git clone https://github.com/SpruceGabriela/CST8504_Assignment1
   cd CST8504_Assignment1
   ```

## 🐳 Run with Docker (recommended)
Requires Docker Desktop.

```bash
docker compose up --build
```

Or, without Compose:

```bash
docker build -t solar_dashboard .
docker run --rm -p 8501:8501 solar_dashboard
```

Open http://localhost:8501. Press `Ctrl+C` to stop.

OR

## 📥 Run locally (without Docker)

1. Create and activate the Conda environment:
   ```bash
   conda create -n ai_techniques_assignment1 python=3.10
   conda activate ai_techniques_assignment1
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dashboard:
   ```bash
   streamlit run app/dashboard.py
   ```

## 🧠 Rebuild the data and train the model
The processed data and the trained model are already in the repository, so this is only needed if the data pipeline changes.

1. Clean the raw data, split it (70/15/15) and save the scaler:
   ```bash
   python src/data_processor.py
   ```
   Creates `data/processed/train.csv`, `validation.csv`, `test.csv` and `models/scaler.pkl`.

2. Train and evaluate the Linear Regression:
   ```bash
   python src/model_trainer.py
   ```
   Creates in `models/`:
   - `linear_regression.pkl`: the trained model, used by the Prediction page
   - `model_metrics.csv`, `test_predictions.csv`, `coefficients.csv`: used by the Model Performance page

Run both commands from the project root. If the data pipeline changes, run both again, so the scaler and the model stay in sync.

---

## 📌 Freezing Dependencies

Whenever you install new packages, update the requirements.txt file so everyone on the team has the same dependencies:

```
pip freeze > requirements.txt
```

---

## 🧹 Code Quality with Ruff

We use Ruff to keep our Python code clean and formatted.

- Check code for errors:
```
ruff check .
```
- Format code automatically:
```  
ruff format .
```

---

## ✅ Running Tests

From the project root:
```bash
python -m pytest -v
```
Use `python -m pytest` (not just `pytest`), so the imports from `src/` work.
The tests check that the trainer runs end to end and that the Predictor returns a valid kWh value and raises clear errors for bad input.

---

## 📝 Commit Messages Guidelines

For the purpose of keeping our Git history organized, we can follow the Conventional Commits specification (https://www.conventionalcommits.org/).
