import csv
import json
import matplotlib.pyplot as plt

def load_data(filename):
    kms = []
    prices = []
    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                kms.append(float(row[0]))
                prices.append(float(row[1]))
    except (FileNotFoundError, ValueError, IndexError) as e:
        print(f"Erreur lors du chargement des données : {e}")
        exit(1)
    return kms, prices

def normalize_data(data):
    min_val = min(data)
    max_val = max(data)
    if max_val == min_val:
        return [0.0] * len(data), min_val, max_val
    return [(x - min_val) / (max_val - min_val) for x in data], min_val, max_val

def denormalize_price(normalized_price, min_price, max_price):
    return normalized_price * (max_price - min_price) + min_price

def train_model(kms, prices):
    learning_rate = 0.42
    iterations = 10000

    theta0 = 0.0
    theta1 = 0.0

    m = float(len(kms))

    for _ in range(iterations):
        sum0 = 0
        sum1 = 0
        for j in range(len(kms)):
            estimated_price = theta0 + (theta1 * kms[j])

            error = estimated_price - prices[j]

            sum0 += error
            sum1 += error * kms[j]

        tmp0 = learning_rate * (1/m) * sum0
        tmp1 = learning_rate * (1/m) * sum1

        theta0 -= tmp0
        theta1 -= tmp1

    return theta0, theta1

def save_parameters(theta0, theta1, min_km, max_km, min_price, max_price):
    params = {
        'theta0': theta0,
        'theta1': theta1,
        'min_km': min_km,
        'max_km': max_km,
        'min_price': min_price,
        'max_price': max_price
    }
    try:
        with open('model_parameters.json', 'w') as f:
            json.dump(params, f, indent=4)
        print("Modèle entraîné et paramètres sauvegardés dans model_parameters.json")
    except IOError as e:
        print(f"Erreur lors de la sauvegarde des paramètres : {e}")

def calculate_r_squared(kms_orig, kms_norm, prices_norm, theta0, theta1):
    m = len(kms_orig)
    mean_price = sum(prices_norm) / m

    tss = sum([(p - mean_price)**2 for p in prices_norm])

    rss = 0
    for i in range(m):
        predicted_price = theta0 + theta1 * kms_norm[i]
        rss += (prices_norm[i] - predicted_price)**2

    if tss == 0:
        return 1.0

    r2 = 1 - (rss / tss)
    print(f"Précision du modèle (R au carré) : {r2:.4f}")
    return r2

def plot_results(kms_orig, prices_orig, theta0, theta1, min_km, max_km, min_price, max_price):
    plt.figure(figsize=(10, 6))

    plt.scatter(kms_orig, prices_orig, label='Données réelles')

    x_line = [min(kms_orig), max(kms_orig)]
    x_line_norm = [(x - min_km) / (max_km - min_km) for x in x_line]

    y_line_norm = [theta0 + theta1 * x for x in x_line_norm]
    y_line = [denormalize_price(y, min_price, max_price) for y in y_line_norm]

    plt.plot(x_line, y_line, color='red', label='Ligne de régression')

    plt.title('Régression Linéaire: Prix des voitures en fonction du kilométrage')
    plt.xlabel('Kilométrage (km)')
    plt.ylabel('Prix (€)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    kms_original, prices_original = load_data('data.csv')

    kms_normalized, min_km, max_km = normalize_data(kms_original)
    prices_normalized, min_price, max_price = normalize_data(prices_original)

    print("Début de l'entraînement du modèle...")
    theta0, theta1 = train_model(kms_normalized, prices_normalized)

    save_parameters(theta0, theta1, min_km, max_km, min_price, max_price)

    calculate_r_squared(kms_original, kms_normalized, prices_normalized, theta0, theta1)

    plot_results(kms_original, prices_original, theta0, theta1, min_km, max_km, min_price, max_price)
