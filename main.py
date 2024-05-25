import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('daten.csv')

# Feature Engineering
# Annahme: Wir wollen nur numerische Features verwenden und die Zufriedenheit als Label
features = ['Budget', 'District', 'City']  # Liste der Features
target = 'Satisfaction'  # Label

# Aufteilen in Features und Label
X = data[features].values
y = data[target].values

# Daten normalisieren
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Daten in Trainings- und Testsets aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Definition der Modellarchitektur
class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Modellinitialisierung
input_dim = X_train.shape[1]
model = Model(input_dim)

# Definition der Verlustfunktion und des Optimierungsalgorithmus
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Konvertierung zu Torch-Tensoren
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Modelltraining
num_epochs = 1000
for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs.squeeze(),
                     y_train_tensor)  # Squeeze wird verwendet, um die Ausgabe in die richtige Form zu bringen
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# Modellbewertung
with torch.no_grad():
    outputs = model(X_test_tensor)
    test_loss = criterion(outputs.squeeze(), y_test_tensor)
    print(f'Test Loss: {test_loss.item()}')

# torch.save(model.state_dict(), 'model.pth')
