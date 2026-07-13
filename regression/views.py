from django.shortcuts import render
import tensorflow as tf
import pandas as pd

# Chargement du modèle
model = tf.keras.models.load_model("models/model.keras")

# Chargement des statistiques
train_stats = pd.read_csv("models/train_stats.csv", index_col=0)

def home(request):

    prediction = None

    if request.method == "POST":

        data = pd.DataFrame([{
            "Cylinders": float(request.POST["cylinders"]),
            "Displacement": float(request.POST["displacement"]),
            "Horsepower": float(request.POST["horsepower"]),
            "Weight": float(request.POST["weight"]),
            "Acceleration": float(request.POST["acceleration"]),
            "Model Year": float(request.POST["year"]),
            "Origin_Europe": float(request.POST["origin_europe"]),
            "Origin_Japan": float(request.POST["origin_japan"]),
            "Origin_USA": float(request.POST["origin_usa"]),
        }])

        # Normalisation avec les mêmes statistiques que pendant l'entraînement
        data = (data - train_stats["mean"]) / train_stats["std"]

        prediction = model.predict(data, verbose=0)[0][0]

    return render(
        request,
        "regression/index.html",
        {"prediction": prediction}
    )