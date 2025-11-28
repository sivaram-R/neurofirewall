import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import pygame
import time
import random
from datetime import datetime
import uuid

# -------------------------------------------------------
# Streamlit setup
# -------------------------------------------------------
st.set_page_config(
    page_title="🧠 Neuro Firewall – Real-Time Brainwave Defense",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <h1 style='text-align:center;'>🧠 Neuro Firewall – AI Brain Defense System</h1>
    <p style='text-align:center;color:gray;'>Real-time EEG analysis with neural activity visualization and adaptive background transitions.</p>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------
# Load model and preprocessors
# -------------------------------------------------------
model = tf.keras.models.load_model("models/neuro_firewall_real.h5")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/label_encoder.pkl")
df = pd.read_csv("data/emotions.csv")
features = df.drop("label", axis=1).values

# -------------------------------------------------------
# Sound feedback
# -------------------------------------------------------
pygame.mixer.init(frequency=44100, size=-16, channels=2)

def play_feedback(level):
    """Short tone feedback for each level."""
    if level == "Safe":
        frequency, duration, volume = 440, 0.25, 0.2
    elif level == "Alert":
        frequency, duration, volume = 880, 0.25, 0.25
    else:
        frequency, duration, volume = 1760, 0.35, 0.3

    sample_rate = 44100
    n_samples = int(round(duration * sample_rate))
    t = np.linspace(0, duration, n_samples, False)
    tone = np.sin(frequency * t * 2 * np.pi)
    stereo_tone = np.column_stack((tone, tone))
    sound_array = (stereo_tone * 32767 * volume).astype(np.int16)
    sound = pygame.sndarray.make_sound(sound_array)
    sound.play()

# -------------------------------------------------------
# EEG generator
# -------------------------------------------------------
def generate_eeg_signal(level):
    mean_vector = np.mean(features, axis=0)
    std_vector = np.std(features, axis=0)
    noise_scale = {"Safe": 0.05, "Alert": 0.15, "Harmful": 0.35}[level]
    eeg = mean_vector + np.random.normal(0, std_vector * noise_scale)
    eeg_scaled = scaler.transform([eeg])
    return eeg_scaled, noise_scale

# -------------------------------------------------------
# Prediction logic
# -------------------------------------------------------
def predict_eeg(eeg_scaled, noise_scale):
    pred = model.predict(eeg_scaled, verbose=0)
    predicted_label = encoder.inverse_transform([np.argmax(pred)])[0]
    confidence = float(np.max(pred))

    if noise_scale < 0.1:
        decision, color, level = "✅ SAFE", "green", "Safe"
    elif noise_scale < 0.25:
        decision, color, level = "⚠️ ALERT", "orange", "Alert"
    else:
        decision, color, level = "🚫 HARMFUL", "red", "Harmful"

    return decision, color, predicted_label, confidence, level

# -------------------------------------------------------
# Neural activation visualization
# -------------------------------------------------------
def draw_dynamic_brain(ax, activity):
    grid_x, grid_y = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-2, 2, 150))
    brain_activity = (
        np.sin(2 * grid_x + time.time() * 2)
        * np.cos(2 * grid_y - time.time())
        + np.exp(-0.2 * (grid_x**2 + grid_y**2))
    ) * (1 + activity * 2)
    cmap = "inferno" if activity > 0.25 else "plasma"
    ax.imshow(brain_activity, cmap=cmap, origin="lower", interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Neural Activity Map", color="white", fontsize=10)

# -------------------------------------------------------
# Background Controller (robust reset)
# -------------------------------------------------------
def reset_background():
    """
    Remove any previous injected bg- style tags and clear container background.
    This JS removes old <style id="bg-..."> elements and forces the container bg to transparent.
    """
    js_reset = """
    <script>
    (function(){
      try {
        // remove any previous style tags with id starting with bg-
        const old = document.querySelectorAll('style[id^="bg-"]');
        old.forEach(n => n.remove());

        // also remove any inline background on the app container
        const appView = document.querySelector('[data-testid="stAppViewContainer"]');
        if (appView) {
            appView.style.backgroundColor = 'rgba(0,0,0,0)';
            appView.style.transition = 'background-color 0.2s linear';
        }
      } catch(e){ console.log("bg reset error", e) }
    })();
    </script>
    """
    st.markdown(js_reset, unsafe_allow_html=True)

def set_background(color_hex, opacity):
    """Set background via a uniquely identified <style> tag so it can be removed next run."""
    css_id = f"bg-{uuid.uuid4().hex}"
    r, g, b = tuple(int(color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    rgba = f"rgba({r},{g},{b},{opacity})"
    st.markdown(
        f"""
        <style id="{css_id}">
        [data-testid="stAppViewContainer"] {{
            background-color: {rgba} !important;
            transition: background-color 1.0s ease-in-out;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------
# Dashboard
# -------------------------------------------------------
def run_live_dashboard():
    col1, col2 = st.columns([2, 1])
    waveform_placeholder = col1.empty()
    brain_placeholder = col1.empty()
    log_area = col2.empty()
    logs = []

    # Randomly select scan state
    level = random.choice(["Safe", "Alert", "Harmful"])
    eeg_scaled, noise_scale = generate_eeg_signal(level)
    play_feedback(level)

    indicator_colors = {
        "Safe": "#00ff00",
        "Alert": "#ff9900",
        "Harmful": "#ff0000",
    }
    color = indicator_colors[level]

    # Reset previous styles and then set transparent tone
    reset_background()
    time.sleep(0.05)  # allow DOM changes to apply
    set_background(color, opacity=0.15)
    st.toast(f"🧠 Initiating {level} scan...", icon="⚡")

    for i in range(100):
        noise = np.random.normal(0, 0.02, eeg_scaled.shape[1])
        new_signal = eeg_scaled[0] + noise

        if i % 10 == 0:
            decision, _, label, confidence, level = predict_eeg(np.array([new_signal]), noise_scale)
            logs.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] {decision} | Emotion: {label} | Confidence: {confidence:.2f}")
            play_feedback(level)

        # EEG waveform
        fig1, ax1 = plt.subplots(figsize=(6, 2))
        ax1.plot(new_signal[:100], color=color, linewidth=2)
        ax1.set_title(f"EEG Signal ({level})", fontsize=10)
        ax1.set_yticks([])
        ax1.set_xlim(0, 100)
        waveform_placeholder.pyplot(fig1)
        plt.close(fig1)

        # Neural brain activity
        fig2, ax2 = plt.subplots(figsize=(3.5, 3), facecolor='black')
        draw_dynamic_brain(ax2, noise_scale)
        brain_placeholder.pyplot(fig2)
        plt.close(fig2)

        # Logs
        with log_area.container():
            st.markdown("### 🧾 Live Threat Log")
            for entry in logs[:10]:
                tag = "🟩" if "SAFE" in entry else "🟧" if "ALERT" in entry else "🟥"
                st.markdown(f"{tag} {entry}")

        time.sleep(0.2)

    # Slightly more opaque at completion (same hue)
    set_background(color, opacity=0.4)
    st.success(f"🧠 Scan complete – Status: {level.upper()}")

# -------------------------------------------------------
# Main Button
# -------------------------------------------------------
if st.button("Start Live EEG Analysis"):
    # ensure previous styles are removed right before run
    reset_background()
    with st.spinner("Initializing brainwave simulation..."):
        run_live_dashboard()
else:
    # keep UI neutral when idle
    reset_background()
    st.info("Click the button to start live EEG scan.")
