# Car Price Prediction System

ECM3440 Software Engineering 2 Assignment.

This project is a proof-of-concept web application that uses machine learning to predict UK used-car prices from vehicle characteristics such as brand, model, year, mileage, transmission, fuel type and engine size.

The system is developed using **Python, Flask, Scikit-Learn, Pandas and NumPy**.

## Machine Learning

The project uses **supervised regression** to predict vehicle prices.

Four regression models were evaluated:

- Linear Regression
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression

The models were compared using **MAE, RMSE, MAPE and R²**. Random Forest produced the strongest overall performance and was selected and tuned for the final system.

The trained model and preprocessing pipeline are saved in:

```text
Saved model/
├── model.pkl
└── preprocessing.pkl
```

The model files are managed using **Git LFS** due to their size.

## Data

The project uses the **100,000 UK Used Car Dataset** from Kaggle, licensed under **CC0: Public Domain**.

Source datasets in `data/` are cleaned, standardised and combined into:

```text
data/database.csv
```

The master dataset contains vehicle features including brand, model, year, transmission, mileage, fuel type, tax, MPG and engine size.

## Setup

Install the required dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Application

```powershell
.\.venv\Scripts\python.exe app.py
```

The Flask application allows users to enter vehicle details and receive an estimated vehicle price.


## Testing

Run the automated tests using:

```powershell
python.exe -m pytest
```

## Limitations

Predictions are indicative estimates based on historical vehicle data and should not be treated as guaranteed valuations. The proof of concept does not currently use live market data or automatically retrain the model.