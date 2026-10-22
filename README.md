# Software-Engineering-2-Assignment

ECM3440 Software Engineering 2 Assignment

## Data and model

The source listings in `data/` are merged and cleaned into `data/database.csv`.
The master dataset includes a `brand` column and uses the shared fields `model`,
`year`, `price`, `transmission`, `mileage`, `fuel_type`, `tax`, `mpg`, and
`engine_size`. Missing optional values are imputed, rows without a usable model,
year, or price are excluded, and duplicate listings are removed.

Rebuild the master dataset and train the model from the repository root:

```powershell
.\.venv\Scripts\python.exe -m model.data_loader
.\.venv\Scripts\python.exe -m model.save_model
```

Run the web app from the repository root:

```powershell
.\.venv\Scripts\python.exe app.py
```

Run tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest
```
