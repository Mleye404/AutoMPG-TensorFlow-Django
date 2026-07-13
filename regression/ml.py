import os
import pandas as pd
import tensorflow as tf

# ==========================
# Chargement du dataset
# ==========================

column_names = [
    "MPG",
    "Cylinders",
    "Displacement",
    "Horsepower",
    "Weight",
    "Acceleration",
    "Model Year",
    "Origin"
]

dataset = pd.read_csv(
    "dataset/auto-mpg.data",
    names=column_names,
    usecols=range(8),
    na_values="?",
    comment="\t",
    sep=r"\s+",
    skipinitialspace=True
)

print("===== Dataset =====")
print(dataset.head())

print("\n===== Valeurs manquantes =====")
print(dataset.isna().sum())

# ==========================
# Nettoyage
# ==========================

dataset = dataset.dropna()

dataset["Origin"] = dataset["Origin"].map({
    1: "USA",
    2: "Europe",
    3: "Japan"
})

dataset = pd.get_dummies(
    dataset,
    columns=["Origin"],
    dtype=int
)

print("\n===== Dataset après nettoyage =====")
print(dataset.head())

print("\nNombre de lignes :", len(dataset))

# ==========================
# Train / Test
# ==========================

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

print("\nTrain :", len(train_dataset))
print("Test :", len(test_dataset))

# ==========================
# Labels
# ==========================

train_labels = train_dataset.pop("MPG")
test_labels = test_dataset.pop("MPG")

# ==========================
# Statistiques
# ==========================

train_stats = train_dataset.describe().transpose()

print("\n===== Statistiques =====")
print(train_stats)

# ==========================
# Normalisation
# ==========================

def normalize(x):
    return (x - train_stats["mean"]) / train_stats["std"]

norm_train_data = normalize(train_dataset)
norm_test_data = normalize(test_dataset)

print("\n===== Données normalisées =====")
print(norm_train_data.head())

# ==========================
# Modèle
# ==========================

def build_model():

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(norm_train_data.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="mse",
        metrics=["mae", "mse"]
    )

    return model

model = build_model()

print("\n===== Résumé du modèle =====")
model.summary()

# ==========================
# Entraînement
# ==========================

history = model.fit(
    norm_train_data,
    train_labels,
    validation_split=0.2,
    epochs=100,
    verbose=1
)

# ==========================
# Evaluation
# ==========================

loss, mae, mse = model.evaluate(
    norm_test_data,
    test_labels,
    verbose=2
)

print("\nErreur moyenne absolue :", mae)

# ==========================
# Sauvegarde
# ==========================

os.makedirs("models", exist_ok=True)

model.save("models/model.keras")

train_stats.to_csv("models/train_stats.csv")

print("\nModèle enregistré : models/model.keras")
print("Statistiques enregistrées : models/train_stats.csv")