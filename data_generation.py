import pandas as pd
import numpy as np

# Anzahl der Beispieldaten
num_samples = 1000

# Beispiel-Daten generieren
np.random.seed(42)  # Für die Reproduzierbarkeit der Ergebnisse
budgets = np.random.randint(500, 3000, num_samples)
districts = np.random.randint(1, 5, num_samples)
cities = np.random.randint(1, 4, num_samples)
satisfactions = np.random.randint(1, 11, num_samples)

# Dictionary mit den generierten Daten erstellen
data = {
    'Budget': budgets,
    'District': districts,
    'City': cities,
    'Satisfaction': satisfactions
}

# DataFrame erstellen
df = pd.DataFrame(data)

# DataFrame in eine CSV-Datei speichern
df.to_csv('daten.csv', index=False)
