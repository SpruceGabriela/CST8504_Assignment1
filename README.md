# CST8504 Assignment 1 Prediction Dashboard

This project is an end-to-end Machine Learning prediction application built for CST8504 Assignment 1. 
It features a complete pipeline: cleaning and exploring messy tabular data, training and tuning supervised learning models (such as Linear Regression or Deep Neural Networks), and real-time predictions through an interactive Streamlit dashboard. The entire application is version-controlled with Git/GitHub and fully containerized using Docker for seamless deployment on any machine.

## 🛠️ Tools & Technologies

- Python 3.10
- Conda (Environment Management)
- Pip (Package Manager)
- Ruff (Linter & Formatter)

---

## 📥 Installation

1. Clone the repository:
   git clone https://github.com/SpruceGabriela/CST8504_Assignment1
   cd /CST8504_Assignment1

2. Create and activate the Conda environment:
   conda create -n ai_techniques_assignment1 python=3.10
   conda activate ai_techniques_assignment1

3. Install dependencies:
   pip install -r requirements.txt

---

## 📌 Freezing Dependencies

Whenever you install new packages, update the requirements.txt file so everyone on the team has the same dependencies:

pip freeze > requirements.txt

---

## 🧹 Code Quality with Ruff

We use Ruff to keep our Python code clean and formatted.

- Check code for errors:
  ruff check .

- Fix formatting and code style automatically:
  ruff check --fix .

---

## 📝 Commit Messages Guidelines

For the purpose of keeping our Git history organized, we can follow the Conventional Commits specification (https://www.conventionalcommits.org/).
