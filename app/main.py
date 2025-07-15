import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

input_window = 10
output_window = 1
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df = pd.read_excel("app/data/One month Data (1).xlsx", skiprows=2)
df.columns = df.columns.map(lambda x: str(x).strip().lower())

df = df.rename(columns={
    'humidity (%)': 'humidity',
    'pyranometer (watts/m2)': 'solar_radiation',
    'rain_fall (mm)': 'rainfall',
    'air_temperature (degc)': 'air_temp',
    '25': 's1_25', '50': 's1_50', '80': 's1_80', '110': 's1_110',
    '25.1': 's2_25', '60': 's2_60', '100': 's2_100', '140': 's2_140',
    '25.2': 's3_25', '60.1': 's3_60', '110.1': 's3_110', '170': 's3_170',
    '25.3': 's4_25', '60.2': 's4_60', '120': 's4_120', '190': 's4_190'
})

df['timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek

df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

required_columns = [
    'air_temp', 'humidity', 'solar_radiation', 'rainfall',
    's1_25', 's1_50', 's1_80', 's1_110',
    's2_25', 's2_60', 's2_100', 's2_140',
    's3_25', 's3_60', 's3_110', 's3_170',
    's4_25', 's4_60', 's4_120', 's4_190',
    'hour', 'dayofweek', 'hour_sin', 'hour_cos']

missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")
df = df[required_columns].dropna()

def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

targets = {name: normalize(df[name].values) for name in required_columns}

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(1)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, feature_size=16, num_layers=4, dropout=0.1):
        super().__init__()
        self.model_type = "Transformer"
        self.input_proj = nn.Linear(1, feature_size)
        self.pos_encoder = PositionalEncoding(feature_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=1, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)

    def forward(self, src):
        src = self.input_proj(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        last = output[-1]
        return self.decoder(last)


def create_inout_sequences(data, input_window, output_window):
    input_seq = []
    target_seq = []
    L = len(data)
    for i in range(L - input_window - output_window):
        x = data[i:i + input_window]
        y = data[i + input_window:i + input_window + output_window]
        input_seq.append(x)
        target_seq.append(y)

    input_seq = np.array(input_seq).reshape(-1, input_window, 1)
    target_seq = np.array(target_seq).reshape(-1, output_window, 1)
    return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target_seq, dtype=torch.float32)

def get_batch(inputs, targets, i, batch_size):
    batch_x = inputs[i:i+batch_size]
    batch_y = targets[i:i+batch_size]
    return batch_x.to(device), batch_y.to(device)

def train_model(model, train_x, train_y, optimizer, criterion, scheduler, epochs=200):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.
        for i in range(0, len(train_x) - batch_size, batch_size):
            optimizer.zero_grad()
            input, target = get_batch(train_x, train_y, i, batch_size)
            input = input.transpose(0, 1)
            output = model(input)
            target = target[:, -1]
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.6f}")

def evaluate_model(model, val_x, val_y):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for i in range(0, len(val_x) - batch_size, batch_size):
            input, target = get_batch(val_x, val_y, i, batch_size)
            input = input.transpose(0, 1)
            output = model(input)
            target = target[:, -1].squeeze().cpu().numpy()
            preds.append(output.cpu().numpy())
            trues.append(target)
    return np.concatenate(preds), np.concatenate(trues)

results = {}
for name, series in targets.items():
    split = int(0.7 * len(series))
    train_series = series[:split]
    val_series = series[split:]

    train_seq, train_labels = create_inout_sequences(train_series, input_window, output_window)
    val_seq, val_labels = create_inout_sequences(val_series, input_window, output_window)

    model = TransformerModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
    criterion = nn.MSELoss()

    print(f"\nTraining on: {name.upper()}")
    train_model(model, train_seq, train_labels, optimizer, criterion, scheduler)

    pred, true = evaluate_model(model, val_seq, val_labels)
    print(f"Sample predictions for {name}: {pred[:10]}")
    print(f"Sample true values: {true[:10]}")

    results[name] = (pred, true)

plt.figure(figsize=(15, 12))
for i, (name, (pred, true)) in enumerate(results.items(), 1):
    plt.subplot(len(results) // 2 + 1, 2, i)
    plt.plot(true, label="Actual", color="red", alpha=0.6)
    plt.plot(pred, label="Forecast", color="blue", linestyle="dashed")
    plt.title(f"{name.upper()} Forecast")
    plt.xlabel("Time Step")
    plt.ylabel(name)
    plt.legend()
plt.tight_layout()
plt.show()
import pickle
with open("app/saved_preds/forecasts.pkl", "wb") as f:
    pickle.dump(results, f)

