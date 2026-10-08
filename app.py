from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

from UserInputHandling.preprocessing_user_input import prepare_user_input


app = Flask(__name__)


# Load the trained model
model = joblib.load("Saved model/model.pkl")

# Load the fitted preprocessing
preprocessing = joblib.load("Saved model/preprocessing.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        # Get values entered by the user
        user_input = pd.DataFrame([{

            "Make": request.form.get("make"),
            "Model": request.form.get("model"),
            "Year": request.form.get("year"),

            "Engine Fuel Type":
                request.form.get("engine_fuel_type"),

            "Engine HP":
                request.form.get("engine_hp"),

            "Engine Cylinders":
                request.form.get("engine_cylinders"),

            "Transmission Type":
                request.form.get("transmission_type"),

            "Driven_Wheels":
                request.form.get("driven_wheels"),

            "Number of Doors":
                request.form.get("number_of_doors"),

            "Market Category":
                request.form.get("market_category"),

            "Vehicle Size":
                request.form.get("vehicle_size"),

            "Vehicle Style":
                request.form.get("vehicle_style"),

            "highway MPG":
                request.form.get("highway_mpg"),

            "city mpg":
                request.form.get("city_mpg"),

            "Popularity":
                request.form.get("popularity")

        }])


        # Prepare the user's input
        user_input = prepare_user_input(user_input)


        # Apply the same preprocessing used when training
        processed_input = preprocessing.transform(
            user_input
        )


        # Make the prediction
        prediction = model.predict(
            processed_input
        )


        # Convert log prediction back to normal price
        predicted_price = np.expm1(
            prediction[0]
        )


        # Display the result
        return render_template(
            "result.html",
            predicted_price=predicted_price
        )


    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)