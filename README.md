
#  NeuroFirewall – AI-Powered Real-Time Brainwave Threat Detection

### **An Intelligent EEG-based Cyber Defense Simulation System**

NeuroFirewall is an **AI-driven real-time EEG (brainwave) analysis system** built using **TensorFlow, Streamlit, and dynamic neural visualizations**.
It simulates brain activity, classifies emotional/threat levels using a neural network model, and provides **live visualization, adaptive background transitions, sound alerts, and threat logs**.

##  **Features**

###  **1. AI EEG Emotion/Threat Classification**

* ML model trained on *emotions.csv*
* Predicts 3 states:

  * **Safe**
  * **Alert**
  * **Harmful**

###  **2. Real-Time EEG Signal Simulation**

* Generates synthetic EEG data based on noise scaling
* Streams dynamically to the UI at runtime

###  **3. Neural Activity Heatmap Visualization**

* Animated brain activity using Matplotlib
* Color-coded activation intensity maps

###  **4. Live Adaptive Background System**

* Background color changes based on threat level:

  * 🟩 Safe → Green
  * 🟧 Alert → Orange
  * 🟥 Harmful → Red
* Auto-clears previous background styles to avoid stacking

###  **5. Audio Feedback Engine**

* Uses **pygame** to generate sound tones for each threat state

###  **6. Real-Time Threat Logging**

* Live timestamped entries of:

  * State (Safe/Alert/Harmful)
  * Predicted emotion
  * Confidence

###  **7. One-Click Live EEG Dashboard**

* Dynamic waveform
* Neural heatmap
* Threat alerts
* Toast notifications
* Smooth background transitions

#  Project Structure

```
NeuroFirewall/
│
├── main.py                 # Streamlit live dashboard UI  :contentReference[oaicite:2]{index=2}
├── train_model.py          # Model training and saving     :contentReference[oaicite:3]{index=3}
│
├── data/
│   └── emotions.csv        # Training dataset
│
├── models/
│   ├── neuro_firewall_real.h5
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
└── README.md
```
#  Dataset Used

The project uses an **emotional EEG dataset**:

* Features = EEG signal values
* Target label = `"label"` column (Safe / Alert / Harmful)

Dataset is loaded in both scripts:
✔ `main.py` (for live simulation)
✔ `train_model.py` (for training)

#  How It Works

## ** 1. Model Training (train_model.py)**

This script:

* Loads dataset
* Encodes labels
* Scales features
* Creates a feedforward neural network
* Trains on EEG data
* Saves:

  * Model → `neuro_firewall_real.h5`
  * Scaler
  * Label encoder

➡ Run:

```bash
python train_model.py
```

This generates everything required for live prediction.


## ** 2. Live EEG Dashboard (main.py)**

The Streamlit UI does the following:

### **Signal Generation**

Simulates EEG based on “level”:

```python
generate_eeg_signal(level)
```

### **Prediction**

Runs inference:

```python
pred = model.predict(eeg_scaled)
```

### **Neural Visualization**

Draws real-time activity map:

```python
draw_dynamic_brain(ax, activity)
```

### **Background Control**

Ensures clean style management:

* `reset_background()`
* `set_background(color, opacity)`

### **Audio Alerts**

Plays tones via pygame mixer.

### **Threat Log**

UI panel for ongoing events.

➡ Run locally:

```bash
streamlit run main.py
```

#  How to Run the Project

### **1. Clone the repo**

```bash
git clone https://github.com/yourusername/neurofirewall.git
cd neurofirewall
```

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

### **3. Train the Model (optional)**

```bash
python train_model.py
```

### **4. Start the Dashboard**

```bash
streamlit run main.py
```

#  Output Preview

### ✔ Live EEG waveform

### ✔ Animated neural brainmap

### ✔ Sound alerts

### ✔ Confidence & emotion

### ✔ Threat logs

### ✔ Smooth background transitions

#  Tech Stack

| Component         | Technology         |
| ----------------- | ------------------ |
| ML Model          | TensorFlow / Keras |
| UI                | Streamlit          |
| Signal Simulation | NumPy              |
| Audio System      | Pygame             |
| Visualization     | Matplotlib         |
| Preprocessing     | Scikit-learn       |
| Model Storage     | Joblib + H5        |

#  Future Enhancements

* Real EEG hardware integration
* Cloud activity storage
* Multi-user dashboard
* Reinforcement learning for adaptive calibration
* API endpoint for external systems

