# pip install streamlit requests numpy pandas plotly scipy

import streamlit as st
import requests
import numpy as np
import pandas as pd
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, spectrogram
from scipy.stats import entropy
from requests.exceptions import RequestException

# --- Page Configuration and Custom CSS ---
st.set_page_config(
    page_title="Comprehensive Parkinson's Movement Analyzer",
    page_icon="🧠",
    layout="wide",
)

# Using the original user-provided CSS
st.markdown("""
<style>
.metric-box {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 20px 16px;
    text-align: center;
    margin-bottom: 16px;
    height: 100%;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    transition: all 0.3s ease;
    display: flex;
    flex-direction: column;
    justify-content: center;
    cursor: default;
}
.metric-box:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 6px 40px rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.metric-box h4 {
    font-size: 15px;
    font-weight: 500;
    margin: 8px 0 4px 0;
    color: #7f8c8d;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-box p {
    font-size: 26px;
    font-weight: 700;
    margin: 0;
    color: #ffffff;
}
.clinical-metric {
    background-color: #1a1d23;
    border-left: 5px solid #4f4f4f;
    padding: 10px 15px;
    border-radius: 6px;
    margin-bottom: 10px;
}
.clinical-metric .label {
    font-size: 14px;
    color: #aaa;
    margin-bottom: 4px;
}
.clinical-metric .value {
    font-size: 20px;
    font-weight: 600;
    color: #fff;
}
[data-baseweb="tab-list"] { background-color: #0e1117; padding: 6px 10px; border-radius: 10px; border: 1px solid #333; }
[data-baseweb="tab"] { background-color: #1a1d23; color: #aaa; border-radius: 8px; padding: 10px 16px; margin-right: 6px; font-weight: 500; transition: all 0.2s ease-in-out; }
[data-baseweb="tab"]:hover { background-color: #2b2f36; color: #ddd; }
[data-baseweb="tab"][aria-selected="true"] { background-color: #4f4f4f; color: white; box-shadow: 0 2px 6px rgba(0,0,0,0.4); }
</style>
""", unsafe_allow_html=True)

# --- Firebase & Secrets Configuration ---
try:
    FIREBASE_URL = "https://handmove-60155-default-rtdb.asia-southeast1.firebasedatabase.app"
    DB_SECRET = "u0lH75jhUHm7pllYTz2sL8OVCdg7BZglaYNG6z7g"
except (AttributeError, FileNotFoundError):
    st.error("Firebase secrets not found. Please create a `.streamlit/secrets.toml` file.")
    st.stop()


# --- Initialize Session State ---
def init_session_state():
    defaults = {
        'connection_successes': 0, 'connection_failures': 0, 'last_latency': 0.0,
        'last_data_rate': 0.0, 'last_conn_status': "Initializing", 'is_running': True,
        'last_seen_id': 0, 'last_id_time': time.time()
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# --- Sidebar ---
with st.sidebar:
    st.title("Dashboard Settings")
    st.session_state.is_running = st.toggle("Enable Auto-Refresh", value=st.session_state.is_running)
    refresh_interval = st.slider("Refresh Interval (seconds)", 2, 15, 5)

    with st.expander("Composite Index Tuning", expanded=True):
        st.info("Adjust the importance of each factor in the main Tremor Index calculation.")
        st.session_state.w_rms = st.slider("RMS Weight", 0.0, 1.0, 0.4, 0.05)
        st.session_state.w_freq = st.slider("Frequency Power (4-8Hz) Weight", 0.0, 1.0, 0.4, 0.05)
        st.session_state.w_jerk = st.slider("Smoothness (1/Jerk) Weight", 0.0, 1.0, 0.2, 0.05)

    with st.expander("Staging Thresholds", expanded=False):
        st.session_state.stage1_idx = st.slider("Stage 1/2 Boundary", 0.0, 1.0, 0.3)
        st.session_state.stage2_idx = st.slider("Stage 2/3 Boundary", 0.0, 1.0, 0.5)
        st.session_state.stage3_idx = st.slider("Stage 3/4 Boundary", 0.0, 1.0, 0.7)

    st.title("Connection Health")
    total = st.session_state.connection_successes + st.session_state.connection_failures
    success_rate = (st.session_state.connection_successes / total * 100) if total > 0 else 100
    st.metric("Last Fetch", st.session_state.last_conn_status)
    st.metric("Latency", f"{st.session_state.last_latency:.2f} s")
    st.metric("Data Rate", f"{st.session_state.last_data_rate:.1f} KB/s")
    st.progress(int(success_rate), text=f"Success Rate: {success_rate:.1f}%")
    if st.button("Reset Stats"):
        st.session_state.connection_successes = 0
        st.session_state.connection_failures = 0
        st.rerun()


# --- Data Fetching ---
def get_data_from_firebase(URL, KEY):
    diagnostics = {'latency': 0, 'data_rate': 0}
    try:
        start_time = time.time()
        id_url = f"{URL}/last_dataset_id.json?auth={KEY}"
        r_id = requests.get(id_url, timeout=3)
        r_id.raise_for_status()
        dataset_id = int(r_id.json())

        data_url = f"{URL}/datasets/{dataset_id}.json?auth={KEY}"
        r_data = requests.get(data_url, timeout=5)
        r_data.raise_for_status()

        latency = time.time() - start_time
        data_size_kb = len(r_data.content) / 1024
        diagnostics.update({'latency': latency, 'data_rate': data_size_kb / latency if latency > 0 else 0})

        st.session_state.connection_successes += 1
        st.session_state.last_conn_status = "Connected"
        return dataset_id, r_data.json(), diagnostics

    except (RequestException, ValueError, TypeError) as e:
        st.session_state.connection_failures += 1
        st.session_state.last_conn_status = f"Failed: {type(e).__name__}"
        return None, None, diagnostics


# --- ADVANCED ANALYSIS FUNCTIONS ---
def perform_advanced_analysis(data):
    if not data or not isinstance(data, list) or len(data) < 20: return None
    df = pd.DataFrame(data)
    if 't' not in df.columns or 'A1' not in df.columns or 'A2' not in df.columns: return None

    df['time_s'] = (df['t'] - df['t'].iloc[0]) / 1000.0
    # Explicitly use A1 for all primary clinical/tremor analysis
    df[['ax1', 'ay1', 'az1']] = pd.DataFrame(df['A1'].tolist(), index=df.index, dtype=np.float64)
    df[['ax2', 'ay2', 'az2']] = pd.DataFrame(df['A2'].tolist(), index=df.index, dtype=np.float64)

    duration = df['time_s'].iloc[-1]
    if duration == 0: return None
    fs = len(df) / duration

    # --- SANITY CHECK: Check if there is any movement in Sensor 1 ---
    # If the standard deviation is tiny, there's no real tremor to analyze.
    if df['ax1'].std() < 1e-4 and df['ay1'].std() < 1e-4 and df['az1'].std() < 1e-4:
        st.warning("Sensor 1 data shows no significant movement. All clinical features will be zero.")
        # Return a zero-out structure to prevent crashes
        zero_metrics = {k: 0 for k in
                        ["rms_tremor", "stage", "sampling_freq", "effectiveness", "rms_jerk", "power_in_band_ratio",
                         "spectral_entropy", "composite_index", "peak_freq", "band_power_3_7_ratio", "sma",
                         "crest_factor", "zcr"]}
        zero_metrics['std_dev_axes'] = {'ax1': 0, 'ay1': 0, 'az1': 0}
        zero_metrics['stage'] = "No Tremor Detected"
        zero_corr = pd.DataFrame(np.identity(3), columns=['ax1', 'ay1', 'az1'], index=['ax1', 'ay1', 'az1'])
        f_spec, t_spec, Sxx = spectrogram(np.zeros(len(df)), fs)  # Dummy spectrogram
        return df, zero_metrics, pd.DataFrame({'Frequency (Hz)': [], 'Power': []}), (f_spec, t_spec, Sxx), zero_corr

    # Calculate total acceleration magnitude for Sensor 1 (raw tremor)
    df['total_mag'] = np.sqrt(df['ax1'] ** 2 + df['ay1'] ** 2 + df['az1'] ** 2)
    # Calculate total acceleration magnitude for Sensor 2 (stabilized)
    df['total_mag_stable'] = np.sqrt(df['ax2'] ** 2 + df['ay2'] ** 2 + df['az2'] ** 2)

    # --- Time-Domain Analysis (on Sensor 1) ---
    rms_tremor = np.sqrt(np.mean(df['total_mag'] ** 2))
    df['jerk'] = np.gradient(df['total_mag'], df['time_s'])
    rms_jerk = np.sqrt(np.mean(df['jerk'] ** 2))

    # --- Frequency-Domain Analysis (on Sensor 1) ---
    N = len(df['total_mag'])
    yf = rfft(df['total_mag'].to_numpy())
    xf = rfftfreq(N, 1 / fs)
    power_spectrum = np.abs(yf) ** 2
    fft_df = pd.DataFrame({'Frequency (Hz)': xf, 'Power': power_spectrum})

    total_power = np.sum(power_spectrum)
    if total_power == 0: total_power = 1e-9  # Avoid division by zero

    # Power in 4-8Hz band for Composite Index
    power_in_band_4_8_mask = (xf >= 4) & (xf <= 8)
    power_in_band_4_8 = np.sum(power_spectrum[power_in_band_4_8_mask])
    power_in_band_ratio_4_8 = power_in_band_4_8 / total_power

    spectral_entropy_val = entropy(power_spectrum / total_power)

    # --- Composite Tremor Index ---
    norm_rms = min(rms_tremor / 4000, 1.0)
    norm_power_ratio = power_in_band_ratio_4_8
    norm_jerk = min(rms_jerk / 100000, 1.0)
    weights = st.session_state
    weight_sum = weights.w_rms + weights.w_freq + weights.w_jerk
    composite_index = (
                                  weights.w_rms * norm_rms + weights.w_freq * norm_power_ratio + weights.w_jerk * norm_jerk) / weight_sum if weight_sum > 0 else 0

    # --- Spectrogram Data (on Sensor 1) ---
    f_spec, t_spec, Sxx = spectrogram(df['total_mag'], fs)

    # --- CLINICAL FEATURE CALCULATIONS (ALL ON SENSOR 1) ---
    # 1. Peak Frequency (FFT)
    peak_freq_idx = np.argmax(power_spectrum)
    peak_freq = xf[peak_freq_idx] if peak_freq_idx < len(xf) else 0

    # 2. Band Power in 3–7 Hz
    power_in_band_3_7_mask = (xf >= 3) & (xf <= 7)
    power_in_band_3_7 = np.sum(power_spectrum[power_in_band_3_7_mask])
    power_in_band_ratio_3_7 = power_in_band_3_7 / total_power

    # 3. Jerk RMS (already calculated as rms_jerk)
    # 4. Spectral Entropy (already calculated as spectral_entropy_val)
    # 5. Signal Magnitude Area (SMA)
    sma = np.sum(np.abs(df['ax1']) + np.abs(df['ay1']) + np.abs(df['az1'])) / fs

    # 6. Standard Deviation (per axis)
    std_dev_axes = df[['ax1', 'ay1', 'az1']].std().to_dict()

    # 7. Crest Factor
    crest_factor = df['total_mag'].max() / rms_tremor if rms_tremor > 0 else 0

    # 8. Zero Crossing Rate (ZCR) - *** CORRECTED CALCULATION ***
    # This now calculates crossings on each raw axis, which is the correct method.
    zcr_x = np.sum(np.abs(np.diff(np.sign(df['ax1'].to_numpy())))) / (2 * duration)
    zcr_y = np.sum(np.abs(np.diff(np.sign(df['ay1'].to_numpy())))) / (2 * duration)
    zcr_z = np.sum(np.abs(np.diff(np.sign(df['az1'].to_numpy())))) / (2 * duration)
    zcr_total = zcr_x + zcr_y + zcr_z  # Total average crossings per second across all axes

    # 9. Cross-Axis Correlation
    correlation_matrix = df[['ax1', 'ay1', 'az1']].corr()

    # 10. Root Mean Square (RMS) (already calculated as rms_tremor)

    # --- Metrics Dictionary ---
    metrics = {
        "rms_tremor": rms_tremor, "stage": classify_stage_by_index(composite_index), "sampling_freq": fs,
        "effectiveness": (1 - (
                    np.sqrt(np.mean(df['total_mag_stable'] ** 2)) / rms_tremor)) * 100 if rms_tremor > 0 else 0,
        "rms_jerk": rms_jerk, "power_in_band_ratio": power_in_band_ratio_4_8 * 100,
        "spectral_entropy": spectral_entropy_val, "composite_index": composite_index,
        # Populated Clinical Metrics
        "peak_freq": peak_freq,
        "band_power_3_7_ratio": power_in_band_ratio_3_7 * 100,
        "sma": sma,
        "std_dev_axes": std_dev_axes,
        "crest_factor": crest_factor,
        "zcr": zcr_total
    }
    return df, metrics, fft_df, (f_spec, t_spec, Sxx), correlation_matrix

def classify_stage_by_index(index):
    if index < st.session_state.stage1_idx:
        return "Stage 0/1 Mild"
    elif index < st.session_state.stage2_idx:
        return "Stage 2 Moderate"
    elif index < st.session_state.stage3_idx:
        return "Stage 3 Severe"
    else:
        return "Stage 4 Critical"


# --- UI & Plotting Helper Functions ---
def create_metric_box(title, value, help_text=""):
    return f"""<div class="metric-box" title="{help_text}"><p>{value}</p><h4>{title}</h4></div>"""


def create_clinical_metric(label, value, help_text=""):
    return f"""<div class="clinical-metric" title="{help_text}">
                   <div class="label">{label}</div>
                   <div class="value">{value}</div>
               </div>"""


def format_stage_with_color(stage_string):
    colors = {"Mild": "green", "Moderate": "orange", "Severe": "#D35400", "Critical": "red"}
    for stage, color in colors.items():
        if stage in stage_string: return f"<span style='color: {color};'>{stage_string}</span>"
    return f"<span style='color: #333;'>{stage_string}</span>"


# --- MAIN DASHBOARD DISPLAY ---
def display_dashboard(df, metrics, fft_df, spec_data, corr_matrix, dataset_id, device_status, last_update_time):
    st.title("Comprehensive Parkinson's Movement Analyzer")

    # --- Top Row: Key Indicators ---
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown("##### Composite Tremor Index")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=metrics['composite_index'], number={'valueformat': '.2f'},
            domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "Severity", 'font': {'size': 16}},
            gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "dimgray"},
                   'steps': [{'range': [0, st.session_state.stage1_idx], 'color': 'lightgreen'},
                             {'range': [st.session_state.stage1_idx, st.session_state.stage2_idx], 'color': 'yellow'},
                             {'range': [st.session_state.stage2_idx, st.session_state.stage3_idx], 'color': 'orange'},
                             {'range': [st.session_state.stage3_idx, 1], 'color': 'red'}]}))
        fig.update_layout(height=200, margin=dict(t=40, b=10, l=10, r=10), font={'color': "white"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### Detailed Metrics")
        g1, g2, g3 = st.columns(3)
        g1.markdown(create_metric_box("Tremor Stage", format_stage_with_color(metrics['stage']),
                                      "Overall severity classification based on the composite index."),
                    unsafe_allow_html=True)
        g2.markdown(create_metric_box("RMS Power", f"{metrics['rms_tremor']:.0f}",
                                      "Root Mean Square: The overall energy or intensity of the hand tremor."),
                    unsafe_allow_html=True)
        g3.markdown(create_metric_box("Stabilizer Effectiveness", f"{metrics['effectiveness']:.1f}%",
                                      "Percentage reduction in tremor RMS power from the raw hand to the stabilized spoon."),
                    unsafe_allow_html=True)
        g4, g5, g6 = st.columns(3)
        g4.markdown(create_metric_box("Power in 4-8Hz", f"{metrics['power_in_band_ratio']:.1f}%",
                                      "Percentage of tremor power in the typical Parkinsonian frequency band (4-8 Hz). Higher is more indicative."),
                    unsafe_allow_html=True)
        g5.markdown(create_metric_box("Spectral Entropy", f"{metrics['spectral_entropy']:.2f}",
                                      "Measures tremor randomness. Lower values suggest a more regular, predictable tremor."),
                    unsafe_allow_html=True)
        g6.markdown(create_metric_box("RMS of Jerk", f"{metrics['rms_jerk'] / 1000:.1f}k",
                                      "Measures movement 'jerkiness' or lack of smoothness. Higher values mean less smooth movement."),
                    unsafe_allow_html=True)

    st.divider()

    # --- Status Bar ---
    info1, info2, info3, info4 = st.columns(4)
    info1.metric("Device Status", device_status)
    info2.metric("Dataset ID", f"{dataset_id}")
    info3.metric("Sample Rate", f"{metrics['sampling_freq']:.1f} Hz")
    info4.metric("Last Update", last_update_time.strftime('%H:%M:%S'))

    # --- TABS FOR DETAILED ANALYSIS ---
    tab_list = ["Movement Overview", "Temporal Frequency", "Movement Dynamics", "Frequency Based",
                "Clinical Features", "Component Level Details", "Raw"]
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tab_list)

    with tab1:  # Movement Overview
        # ... (code is unchanged)
        st.subheader("Raw Hand vs. Stabilized Spoon Movement (Total Magnitude)")
        st.markdown(
            "This chart compares the raw hand tremor (Sensor 1) with the movement of the stabilized spoon (Sensor 2). Effective stabilization should show a significantly smaller amplitude for Sensor 2.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['time_s'], y=df['total_mag'], mode='lines', name='Raw Hand (Sensor 1)',
                                 line=dict(color='orange')))
        fig.add_trace(
            go.Scatter(x=df['time_s'], y=df['total_mag_stable'], mode='lines', name='Stabilized Spoon (Sensor 2)',
                       line=dict(color='cyan', dash='dash')))
        fig.update_layout(xaxis_title="Time (s)", yaxis_title="Acceleration Magnitude (m/s^2)", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:  # Temporal-Frequency Analysis
        st.subheader("Spectrogram: Tremor Frequency over Time")
        st.info(
            "The spectrogram shows the intensity of different tremor frequencies as they change over the duration of the recording. "
            "Bright horizontal bands in the 3–7 Hz range indicate a persistent Parkinsonian tremor.")

        f_spec, t_spec, Sxx = spec_data
        power_dB = 10 * np.log10(Sxx + 1e-9)  # Convert to dB scale for clarity

        fig = px.imshow(
            power_dB,
            x=t_spec,
            y=f_spec,
            aspect='auto',
            labels=dict(x="Time (s)", y="Frequency (Hz)", color="Power (dB)"),
            title="Tremor Spectrogram",
            color_continuous_scale='plasma',  # Use vibrant colormap: options = 'viridis', 'plasma', 'magma', 'turbo'
            origin='lower'
        )

        fig.update_layout(
            coloraxis_colorbar=dict(
                title="Power (dB)",
                ticks="outside",
                ticklen=5,
                thickness=15
            ),
            margin=dict(l=50, r=50, t=50, b=50),
            font=dict(size=12)
        )

        fig.update_xaxes(title_font=dict(size=14), tickfont=dict(size=12))
        fig.update_yaxes(range=[0, 20], title_font=dict(size=14), tickfont=dict(size=12))

        st.plotly_chart(fig, use_container_width=True)

    with tab3:  # Movement Dynamics
        # ... (code is unchanged)
        st.subheader("Advanced Movement Dynamics")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("**Poincaré Plot: Tremor Variability**")
            st.info(
                "Plots each acceleration value against the next one. A tight, elongated ellipse indicates a highly regular and predictable tremor. A dispersed, circular cloud suggests random or chaotic movement.")
            poincare_df = pd.DataFrame({'a_t': df['total_mag'][:-1], 'a_t+1': df['total_mag'][1:]})
            fig = px.scatter(poincare_df, x='a_t', y='a_t+1', opacity=0.6,
                             labels={'a_t': 'Acceleration at Time t', 'a_t+1': 'Acceleration at Time t+1'},
                             template="plotly_dark")
            fig.update_traces(marker=dict(size=5))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("**3D Movement Trajectory**")
            st.info(
                "Visualizes the path of the tremor in 3D space. This helps identify if the tremor has a primary direction (linear shape) or is more rotational (circular/elliptical shape).")
            fig = px.scatter_3d(df, x='ax1', y='ay1', z='az1', color='time_s',
                                labels={'ax1': 'X-axis', 'ay1': 'Y-axis', 'az1': 'Z-axis'}, template="plotly_dark",
                                opacity=0.7)
            fig.update_traces(marker=dict(size=3))
            st.plotly_chart(fig, use_container_width=True)

    with tab4:  # Frequency Deep-Dive
        # ... (code is unchanged)
        st.subheader("Frequency Spectrum Analysis (FFT)")
        st.info(
            "This breaks down the entire movement signal into its constituent frequencies. A large, sharp peak in the 3-7 Hz range is a classic signature of Parkinsonian tremor.")
        fig = px.bar(fft_df, x='Frequency (Hz)', y='Power', template="seaborn", log_y=True)
        fig.add_vrect(x0=3, x1=7, fillcolor="red", opacity=0.25, line_width=0,
                      annotation_text="Parkinson's Band (3-7 Hz)")
        fig.update_xaxes(range=[0, 25])
        st.plotly_chart(fig, use_container_width=True)

    # --- NEW TAB: CLINICAL FEATURE ANALYSIS ---
    with tab5:
        st.subheader("Clinical Feature Analysis")
        st.info(
            "This section provides a deeper look into specific biomarkers and mathematical features used in clinical research to characterize tremors.")

        # --- Top Row: Radar Chart and Key Metrics ---
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**Tremor Feature Fingerprint**")
            st.markdown(
                "This radar chart provides a multi-dimensional profile of the tremor. A larger shape indicates a more pronounced tremor across various domains.",
                unsafe_allow_html=True)

            # Normalize features for the radar chart on a 0-1 scale based on typical maximums
            categories = ['Intensity (RMS)', 'Roughness (Jerk)', 'PD-Band Power (3-7Hz)', 'Irregularity (Entropy)',
                          'Spikiness (Crest)']
            norm_rms = min(metrics['rms_tremor'] / 5000, 1.0)
            norm_jerk = min(metrics['rms_jerk'] / 150000, 1.0)
            norm_band_power = metrics['band_power_3_7_ratio'] / 100.0
            norm_entropy = min(metrics['spectral_entropy'] / 7, 1.0)  # Theoretical max is ~log(N)
            norm_crest = min((metrics['crest_factor'] - 1) / 10, 1.0)  # Crest factor starts at 1

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[norm_rms, norm_jerk, norm_band_power, norm_entropy, norm_crest],
                theta=categories,
                fill='toself',
                name='Tremor Profile'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False, template="plotly_dark",
                margin=dict(t=40, b=20, l=40, r=40)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Core Biomarkers**")
            st.markdown(create_clinical_metric("Peak Frequency", f"{metrics['peak_freq']:.2f} Hz",
                                               "The single most dominant frequency in the tremor. Parkinson's is typically 3-7 Hz."),
                        unsafe_allow_html=True)
            st.markdown(create_clinical_metric("Power in 3-7Hz Band", f"{metrics['band_power_3_7_ratio']:.1f}%",
                                               "Measures how much of the tremor's energy is concentrated in the typical Parkinsonian frequency range."),
                        unsafe_allow_html=True)
            st.markdown(create_clinical_metric("Crest Factor", f"{metrics['crest_factor']:.2f}",
                                               "Ratio of peak to RMS power. Higher values indicate spiky, non-sinusoidal tremors."),
                        unsafe_allow_html=True)
            st.markdown(create_clinical_metric("Zero-Crossing Rate", f"{metrics['zcr']:.2f} Hz",
                                               "Indicates signal oscillation frequency. Often high in tremors."),
                        unsafe_allow_html=True)
            st.markdown(create_clinical_metric("Signal Magnitude Area", f"{metrics['sma']:.1f}",
                                               "Cumulative measure of movement intensity over the entire recording period."),
                        unsafe_allow_html=True)

        st.divider()

        # --- Bottom Row: Correlation and Variability ---
        col3, col4 = st.columns(2, gap="large")
        with col3:
            st.markdown("**Cross-Axis Correlation**")
            st.info(
                "This heatmap shows the correlation between movements on the X, Y, and Z axes. High correlation (bright squares) can indicate a planar or directional tremor, while low correlation (dark squares) suggests more complex, rotational movement.")
            fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                            color_continuous_scale='RdBu_r', range_color=[-1, 1],
                            labels=dict(color="Correlation"))
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            st.markdown("**Per-Axis Movement Variability (Std. Dev.)**")
            st.info(
                "This bar chart shows the standard deviation of acceleration for each axis. It reveals if the tremor is more pronounced in a specific direction (e.g., up-down vs. side-to-side).")
            std_df = pd.DataFrame(list(metrics['std_dev_axes'].items()), columns=['Axis', 'Standard Deviation'])
            std_df['Axis'] = std_df['Axis'].map({'ax1': 'X-axis', 'ay1': 'Y-axis', 'az1': 'Z-axis'})
            fig = px.bar(std_df, x='Axis', y='Standard Deviation', color='Axis',
                         template="plotly_dark", text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

    with tab6:  # Component-Level Details
        # ... (code is unchanged)
        st.subheader("Per-Axis Time Series Analysis")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("##### Raw Hand Movements (Per Axis)")
            st.line_chart(df.rename(columns={'ax1': 'X', 'ay1': 'Y', 'az1': 'Z'}).set_index('time_s')[['X', 'Y', 'Z']])
        with c2:
            st.markdown("##### Stabilized Spoon Movements (Per Axis)")
            st.line_chart(df.rename(columns={'ax2': 'X', 'ay2': 'Y', 'az2': 'Z'}).set_index('time_s')[['X', 'Y', 'Z']])

    with tab7:  # Raw Data Table
        # ... (code is unchanged)
        st.subheader("Raw Analytical Data Table")
        display_cols = ['time_s', 'ax1', 'ay1', 'az1', 'ax2', 'ay2', 'az2', 'total_mag', 'jerk']
        st.dataframe(df[display_cols].style.format("{:.3f}"), use_container_width=True)


# --- Main Application Logic ---
placeholder = st.empty()

dataset_id, raw_data, diagnostics = get_data_from_firebase(URL=FIREBASE_URL, KEY=DB_SECRET)
st.session_state.last_latency, st.session_state.last_data_rate = diagnostics['latency'], diagnostics['data_rate']

if dataset_id is not None:
    if dataset_id != st.session_state.last_seen_id:
        device_status = "🟢 Online"
        st.session_state.last_seen_id = dataset_id
        st.session_state.last_id_time = time.time()
    elif time.time() - st.session_state.last_id_time > (refresh_interval * 4):
        device_status = "🔴 Offline"
    else:
        device_status = "🟡 Stale"

    processed_result = perform_advanced_analysis(raw_data)
    if processed_result:
        df, metrics, fft_df, spec_data, corr_matrix = processed_result
        placeholder.empty()
        display_dashboard(df, metrics, fft_df, spec_data, corr_matrix, dataset_id, device_status, datetime.now())
    else:
        placeholder.warning("Received data is invalid or sample size is too small. Waiting for new data...")
else:
    placeholder.error("Could not retrieve data from Firebase. Check connection and secrets. Retrying...")

# --- Footer ---
st.markdown("""
    <div style='
        background-color: #0e1117; color: #4f4f4f; text-align: center; padding: 15px;
        font-size: 14px; margin-top: 50px; width: 100%; border-top: 1px solid #4f4f4f;
    '>
        <b>Project Name:</b> Vibration analyzed smart glove to aid Parkinson's patient hand tremor with postural stability<br>
        <b>22LE1-035</b> S.A.P.U.Hemachandra | <b>22LE2-082</b> I.H.C.Udayanga<br>
        <b>Internal Passed</b> | <b>Group number:</b> B 07-18<br>
        <b>Supervisor:</b> Mr. Nuwan Attanayake
    </div>
""", unsafe_allow_html=True)

# --- Auto-Refresh Logic ---
if st.session_state.is_running:
    time.sleep(refresh_interval)
    st.rerun()
else:
    placeholder.info("Auto-refresh is paused. Enable it in the sidebar to see live data.")