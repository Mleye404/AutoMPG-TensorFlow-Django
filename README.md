# 🚗 Auto MPG Predictor

Projet réalisé dans le cadre d'un TP de Machine Learning.

Cette application utilise **TensorFlow** pour entraîner un modèle de régression permettant de prédire la consommation de carburant (MPG) d'un véhicule à partir de ses caractéristiques techniques. Le modèle est ensuite intégré dans une application web développée avec **Django**.

---

## 📌 Fonctionnalités

- Chargement du dataset Auto MPG
- Nettoyage et préparation des données
- Encodage des variables catégorielles
- Normalisation des données
- Entraînement d'un modèle de régression avec TensorFlow
- Évaluation du modèle
- Sauvegarde du modèle entraîné
- Interface web Django permettant de réaliser des prédictions

---

## 🛠 Technologies utilisées

- Python 3
- TensorFlow
- Keras
- Pandas
- NumPy
- Django
- HTML / CSS

---

## 📂 Structure du projet

```
AutoMPG_Project/
│
├── config/
├── dataset/
│   └── auto-mpg.data
├── models/
│   ├── model.keras
│   └── train_stats.csv
├── regression/
│   ├── ml.py
│   ├── views.py
│   ├── urls.py
│   └── templates/
├── manage.py
└── requirements.txt
```

---

## 🚀 Installation

Créer un environnement virtuel :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

## ▶️ Entraîner le modèle

```bash
python regression/ml.py
```

Le modèle entraîné est enregistré dans :

```
models/model.keras
```

---

## 🌐 Lancer l'application Django

```bash
python manage.py migrate
python manage.py runserver
```

Puis ouvrir :

```
http://127.0.0.1:8000
```

---

## 📊 Dataset

Le projet utilise le dataset **Auto MPG** de l'UCI Machine Learning Repository.

Source :

https://archive.ics.uci.edu/ml/datasets/auto+mpg

---

## 📷 Aperçu

L'application permet à l'utilisateur de saisir les caractéristiques d'un véhicule et d'obtenir une estimation de sa consommation (MPG) grâce au modèle TensorFlow.

---

## 👨‍💻 Auteur

**Mouhamadou Leye**

Master 1 – Ingénierie Logicielle et Intelligence Artificielle (MILIA)

École Polytechnique de Thiès (EPT)

2026
