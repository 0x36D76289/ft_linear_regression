import json

def load_parameters(filename):
    try:
        with open(filename, 'r') as f:
            params = json.load(f)
            return params
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Erreur : Le fichier '{filename}' est introuvable ou corrompu.\nIl faut d'abord entraîner le modèle en exécutant 'train.py'.")
        exit(1)

def predict_price(mileage, params):
    theta0 = params['theta0']
    theta1 = params['theta1']
    min_km = params['min_km']
    max_km = params['max_km']
    min_price = params['min_price']
    max_price = params['max_price']

    if max_km == min_km:
        normalized_km = 0.0
    else:
        normalized_km = (mileage - min_km) / (max_km - min_km)

    normalized_price = theta0 + (theta1 * normalized_km)
    predicted_price = normalized_price * (max_price - min_price) + min_price

    return predicted_price

if __name__ == "__main__":
    parameters = load_parameters('model_parameters.json')

    try:
        input_mileage = float(input("Veuillez entrer un kilométrage : "))
    except ValueError:
        print("Erreur : Veuillez entrer un nombre valide.")
        exit(1)

    estimated_price = predict_price(input_mileage, parameters)

    print(f"Pour un kilométrage de {int(input_mileage)} km, le prix estimé est de : {estimated_price:.2f} €")
