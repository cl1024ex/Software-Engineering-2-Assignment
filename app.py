from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from UserInputHandling.preprocessing_user_input import BRANDS, YEAR_MAX, YEAR_MIN, UserInputValidationError, prepare_user_input


app = Flask(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent


# Load the trained model
model = joblib.load(PROJECT_ROOT / "Saved model" / "model.pkl")

# Load the fitted preprocessing
preprocessing = joblib.load(PROJECT_ROOT / "Saved model" / "preprocessing.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        # Get values entered by the user
        user_input = pd.DataFrame([{

            "brand": request.form.get("brand", ""),
            "model": request.form.get("model", ""),
            "year": request.form.get("year", ""),
            "transmission": request.form.get("transmission", ""),
            "mileage": request.form.get("mileage", ""),
            "fuel_type": request.form.get("fuel_type", ""),
            "tax": request.form.get("tax", ""),
            "mpg": request.form.get("mpg", ""),
            "engine_size": request.form.get("engine_size", "")
        }])

        try:
            user_input = prepare_user_input(user_input)
            processed_input = preprocessing.transform(user_input)
        except UserInputValidationError as error:
            input_error = str(error)
            return render_template(
                "home.html",
                input_error=input_error,
                brands=BRANDS,
                min_year=YEAR_MIN,
                max_year=YEAR_MAX
            ), 400
        except (TypeError, ValueError, OverflowError):
            return render_template(
                "home.html",
                input_error=(
                    "Some car details could not be processed. "
                    "Check the values and try again."
                ),
                brands=BRANDS,
                min_year=YEAR_MIN,
                max_year=YEAR_MAX
            ), 400

        prediction = model.predict(processed_input)


        # Convert log prediction back to normal price
        predicted_price = np.expm1(
            prediction[0]
        )


        # Display the result
        return render_template(
            "result.html",
            predicted_price=predicted_price
        )


    return render_template(
        "home.html",
        brands=BRANDS,
        min_year=YEAR_MIN,
        max_year=YEAR_MAX
    )


if __name__ == "__main__":
    app.run(debug=True)