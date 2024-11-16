# Diabetes Prediction using Machine Learning

This project demonstrates the use of machine learning models to predict diabetes progression using the Diabetes Dataset from scikit-learn. 

---

## **Project Overview**
- **Dataset**: The Diabetes dataset provided by `scikit-learn`.
- **Objective**: Predict the progression of diabetes in patients based on various diagnostic measures.
- **Tools**: Python, scikit-learn, pandas, matplotlib, seaborn.

---

## **Dataset Description**
The Diabetes dataset is a built-in dataset from `scikit-learn` that contains 442 samples of diabetes patients. The features include age, sex, BMI, blood pressure, and six blood serum measurements.

- **Features**:
  - Age, sex, BMI, blood pressure, and six blood serum measurements.
- **Target**:
  - A quantitative measure of diabetes progression one year after baseline.

---

## **Setup Instructions**
1. Clone this repository:
   git clone https://github.com/<your-username>/healthcare-predictive-models.git](https://github.com/rtdatasci/EHR.git)
   cd EHR

2. Set up a virtual environment and install dependencies:

bash
Copy code
python -m venv venv
source venv/bin/activate  # For Mac/Linux
.\venv\Scripts\activate   # For Windows
pip install -r requirements.txt

3. Run the main script
python main.py



Features or Diagnostic measures from the dataset:
Diagnostic Measures (Features):
Age: The age of the patient (normalized).
Sex: Gender of the patient (encoded as a normalized numerical value).
BMI: Body Mass Index, a measure of body fat based on height and weight (normalized).
BP: Average blood pressure (normalized).
S1: Total serum cholesterol.
S2: Low-density lipoproteins (LDL).
S3: High-density lipoproteins (HDL).
S4: Total triglycerides.
S5: Lamotrigine levels in serum.
S6: Blood sugar levels.

Organization:
EHR/
│
├── data/                 # Folder for storing datasets
│   ├── raw/              # Raw datasets
│   ├── processed/        # Preprocessed datasets
│
├── notebooks/            # Jupyter notebooks for EDA and modeling
│
├── src/                  # Source code for the project
│   ├── __init__.py       # Initialize Python package
│   ├── preprocessing.py  # Data cleaning and preprocessing scripts
│   ├── train_model.py    # Scripts for training models
│   ├── evaluate.py       # Scripts for evaluating models
│
├── tests/                # Test scripts for validating the code
│   ├── test_preprocessing.py
│   ├── test_train_model.py
│
├── reports/              # Generated reports
│   ├── figures/          # Visualizations
│
├── README.md             # Project documentation
├── requirements.txt      # List of dependencies
├── .gitignore            # Ignore unnecessary files
├── LICENSE               # Licensing information
└── main.py               # Main script to run the project
