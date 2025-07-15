# 🛣️ Pavement Risk Forecasting using Transformers

A deep learning project that uses Transformer-based time series forecasting to predict asphalt pavement conditions and generate risk reports for smart road maintenance systems.

---

## 📌 Overview

This project forecasts future values of key road surface and subsurface parameters (e.g., air temperature, rainfall, S1_25, S4_190, etc.) using a custom Transformer neural network. The goal is to detect potential pavement risks — such as:

- 🔥 **Overheating**
- ❄️ **Slipperiness**
- 🪨 **Cracking Risk**
- ⚠️ **Subgrade Instability**

A final risk report is generated with timestamps and alert messages to enable **proactive and intelligent road maintenance**.

---

## 🧠 Model

- Built from scratch using **PyTorch**
- **Sliding window** approach for sequence-to-one forecasting
- Includes **positional encoding** and multi-layer Transformer encoder
- Trained individually on multiple time-series features

---

## 📁 Dataset

- Input: Excel file with one month of weather + sensor data
- Includes:
  - `air_temp`, `humidity`, `rainfall`, `solar_radiation`
  - Subsurface sensor readings: `s1_25`, `s2_60`, `s4_190`, etc.
- Features engineered:
  - `hour`, `dayofweek`, `hour_sin`, `hour_cos`
- Normalized between 0–1

> 📍 Path: `app/data/One month Data (1).xlsx`

---

