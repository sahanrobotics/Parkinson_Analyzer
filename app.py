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
from scipy.signal import spectrogram
from scipy.stats import entropy
from requests.exceptions import RequestException

# --- Page Configuration and Custom CSS for Dashboard ---
st.set_page_config(
    page_title="Advanced Parkinson's Movement Analyzer",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
/* Dashboard Styles (Dark Theme) */
.metric-box {
    background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 20px; padding: 20px 16px; text-align: center; margin-bottom: 16px;
    height: 100%; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1); backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px); transition: all 0.3s ease; display: flex;
    flex-direction: column; justify-content: center; cursor: default;
}
.metric-box:hover { transform: translateY(-3px) scale(1.02); box-shadow: 0 6px 40px rgba(0, 0, 0, 0.2); border: 1px solid rgba(255, 255, 255, 0.1); }
.metric-box h4 { font-size: 15px; font-weight: 500; margin: 8px 0 4px 0; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-box p { font-size: 26px; font-weight: 700; margin: 0; color: #ffffff; }
.clinical-metric { background-color: #1a1d23; border-left: 5px solid #4f4f4f; padding: 10px 15px; border-radius: 6px; margin-bottom: 10px; }
.clinical-metric .label { font-size: 14px; color: #aaa; margin-bottom: 4px; }
.clinical-metric .value { font-size: 20px; font-weight: 600; color: #fff; }
[data-baseweb="tab-list"] { background-color: #0e1117; padding: 6px 10px; border-radius: 10px; border: 1px solid #333; }
[data-basweeb="tab"] { background-color: #1a1d23; color: #aaa; border-radius: 8px; padding: 10px 16px; margin-right: 6px; font-weight: 500; transition: all 0.2s ease-in-out; }
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
        'last_seen_id': 0, 'last_id_time': time.time(), 'device_status': "Initializing",
        'live_dataset_id': None, 'live_raw_data': None,
        'mode': 'Live', 'patient_id_input': 'P001',
        'recordings_list': None, 'selected_recording_id': None,
        'full_playback_df': None, 'current_window_start': 0.0,
        'total_duration': 0.0,
        'current_window_analysis': None,
        'is_recording': False, 'recorded_data_buffer': [], 'last_recorded_id': -1,
        'viewing_report': False,
        'report_data': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# --- DATA FETCHING & SAVING FUNCTIONS (UNCHANGED) ---
def get_live_data_from_firebase(URL, KEY):
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


def save_recording_to_firebase(patient_id, source_id, data, URL, KEY):
    if not patient_id or not data:
        st.toast("Error: Missing Patient ID or no data in buffer to save.", icon="❌")
        return
    try:
        recording_id = f"rec_{int(time.time())}_{patient_id}"
        payload = {"patient_id": patient_id, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   "source_dataset_id": source_id, "data": data}
        url = f"{URL}/recordings/{recording_id}.json?auth={KEY}"
        r = requests.put(url, json=payload, timeout=15)
        r.raise_for_status()
        st.toast(f"Recording saved for Patient {patient_id}!", icon="💾")
        get_recordings_list.clear()
    except RequestException as e:
        st.error(f"Failed to save recording due to a network error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred while saving the recording: {e}")


@st.cache_data(ttl=60)
def get_recordings_list(URL, KEY):
    try:
        url = f"{URL}/recordings.json?auth={KEY}"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        full_data = r.json()
        if not full_data: return {}
        recordings = {
            rec_id: {"patient_id": details.get("patient_id", "N/A"), "timestamp": details.get("timestamp", "N/A")} for
            rec_id, details in full_data.items() if isinstance(details, dict)}
        return recordings
    except (RequestException, ValueError, TypeError, KeyError) as e:
        st.error(f"Could not fetch or parse recordings list: {e}")
        return {}


@st.cache_data(ttl=3600)
def get_specific_recording(recording_id, URL, KEY):
    try:
        url = f"{URL}/recordings/{recording_id}.json?auth={KEY}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except (RequestException, ValueError) as e:
        st.error(f"Failed to load or parse recording {recording_id}: {e}")
        return None


def delete_recording_from_firebase(recording_id, URL, KEY):
    try:
        url = f"{URL}/recordings/{recording_id}.json?auth={KEY}"
        r = requests.delete(url, timeout=10)
        r.raise_for_status()
        st.toast(f"Recording {recording_id} deleted successfully.", icon="🗑️")
        return True
    except RequestException as e:
        st.error(f"Failed to delete recording: {e}")
        return False


# --- ADVANCED ANALYSIS FUNCTION (UNCHANGED) ---
def perform_advanced_analysis(data):
    if not data or not isinstance(data, list) or len(data) < 20: return None
    try:
        df = pd.DataFrame(data)
        if 't' not in df.columns or 'A1' not in df.columns or 'A2' not in df.columns: return None
        df['time_s'] = (df['t'] - df['t'].iloc[0]) / 1000.0
        df[['ax1', 'ay1', 'az1']] = pd.DataFrame(df['A1'].tolist(), index=df.index, dtype=np.float64)
        df[['ax2', 'ay2', 'az2']] = pd.DataFrame(df['A2'].tolist(), index=df.index, dtype=np.float64)
        duration = df['time_s'].iloc[-1] - df['time_s'].iloc[0]
        if duration <= 0: return None
        fs = len(df) / duration
        if df['ax1'].std() < 1e-4 and df['ay1'].std() < 1e-4 and df['az1'].std() < 1e-4:
            st.warning("Sensor 1 data shows no significant movement. All clinical features will be zero.")
            zero_metrics = {k: 0 for k in
                            ["rms_tremor", "stage", "sampling_freq", "effectiveness", "rms_jerk", "power_in_band_ratio",
                             "spectral_entropy", "composite_index", "peak_freq", "band_power_3_7_ratio", "sma",
                             "crest_factor", "zcr"]}
            zero_metrics['std_dev_axes'] = {'ax1': 0, 'ay1': 0, 'az1': 0};
            zero_metrics['stage'] = "No Tremor Detected"
            zero_corr = pd.DataFrame(np.identity(3), columns=['ax1', 'ay1', 'az1'], index=['ax1', 'ay1', 'az1'])
            f_spec, t_spec, Sxx = spectrogram(np.zeros(len(df)), fs)
            return df, zero_metrics, pd.DataFrame({'Frequency (Hz)': [], 'Power': []}), (f_spec, t_spec, Sxx), zero_corr
        df['total_mag'] = np.sqrt(df['ax1'] ** 2 + df['ay1'] ** 2 + df['az1'] ** 2)
        df['total_mag_stable'] = np.sqrt(df['ax2'] ** 2 + df['ay2'] ** 2 + df['az2'] ** 2)
        rms_tremor = np.sqrt(np.mean(df['total_mag'] ** 2))
        df['jerk'] = np.gradient(df['total_mag'], df['time_s'])
        rms_jerk = np.sqrt(np.mean(df['jerk'] ** 2))
        N = len(df['total_mag']);
        yf = rfft(df['total_mag'].to_numpy());
        xf = rfftfreq(N, 1 / fs)
        power_spectrum = np.abs(yf) ** 2;
        fft_df = pd.DataFrame({'Frequency (Hz)': xf, 'Power': power_spectrum})
        total_power = np.sum(power_spectrum)
        if total_power == 0: total_power = 1e-9
        power_in_band_4_8_mask = (xf >= 4) & (xf <= 8)
        power_in_band_4_8 = np.sum(power_spectrum[power_in_band_4_8_mask]);
        power_in_band_ratio_4_8 = power_in_band_4_8 / total_power
        spectral_entropy_val = entropy(power_spectrum / total_power)
        norm_rms = min(rms_tremor / 4000, 1.0);
        norm_power_ratio = power_in_band_ratio_4_8;
        norm_jerk = min(rms_jerk / 100000, 1.0)
        weights = st.session_state;
        weight_sum = weights.w_rms + weights.w_freq + weights.w_jerk
        composite_index = (
                                  weights.w_rms * norm_rms + weights.w_freq * norm_power_ratio + weights.w_jerk * norm_jerk) / weight_sum if weight_sum > 0 else 0
        f_spec, t_spec, Sxx = spectrogram(df['total_mag'], fs);
        peak_freq_idx = np.argmax(power_spectrum)
        peak_freq = xf[peak_freq_idx] if peak_freq_idx < len(xf) else 0
        power_in_band_3_7_mask = (xf >= 3) & (xf <= 7);
        power_in_band_3_7 = np.sum(power_spectrum[power_in_band_3_7_mask])
        power_in_band_ratio_3_7 = power_in_band_3_7 / total_power
        sma = np.sum(np.abs(df['ax1']) + np.abs(df['ay1']) + np.abs(df['az1'])) / fs
        std_dev_axes = df[['ax1', 'ay1', 'az1']].std().to_dict()
        crest_factor = df['total_mag'].max() / rms_tremor if rms_tremor > 0 else 0
        zcr_x = np.sum(np.abs(np.diff(np.sign(df['ax1'].to_numpy())))) / (2 * duration);
        zcr_y = np.sum(np.abs(np.diff(np.sign(df['ay1'].to_numpy())))) / (2 * duration)
        zcr_z = np.sum(np.abs(np.diff(np.sign(df['az1'].to_numpy())))) / (2 * duration);
        zcr_total = zcr_x + zcr_y + zcr_z
        correlation_matrix = df[['ax1', 'ay1', 'az1']].corr()
        metrics = {"rms_tremor": rms_tremor, "stage": classify_stage_by_index(composite_index), "sampling_freq": fs,
                   "effectiveness": (1 - (
                           np.sqrt(np.mean(df['total_mag_stable'] ** 2)) / rms_tremor)) * 100 if rms_tremor > 0 else 0,
                   "rms_jerk": rms_jerk, "power_in_band_ratio": power_in_band_ratio_4_8 * 100,
                   "spectral_entropy": spectral_entropy_val, "composite_index": composite_index, "peak_freq": peak_freq,
                   "band_power_3_7_ratio": power_in_band_ratio_3_7 * 100, "sma": sma, "std_dev_axes": std_dev_axes,
                   "crest_factor": crest_factor, "zcr": zcr_total, "duration": duration}
        return df, metrics, fft_df, (f_spec, t_spec, Sxx), correlation_matrix
    except Exception as e:
        st.warning(f"Data analysis failed. This can happen with incomplete or corrupt data segments. Error: {e}")
        return None


# --- HELPER & UI FUNCTIONS (UNCHANGED) ---
def classify_stage_by_index(index):
    if index < st.session_state.stage1_idx:
        return "Stage 0/1 Mild"
    elif index < st.session_state.stage2_idx:
        return "Stage 2 Moderate"
    elif index < st.session_state.stage3_idx:
        return "Stage 3 Severe"
    else:
        return "Stage 4 Critical"


def create_metric_box(title, value,
                      help_text=""): return f"""<div class="metric-box" title="{help_text}"><p>{value}</p><h4>{title}</h4></div>"""


def create_clinical_metric(label, value,
                           help_text=""): return f"""<div class="clinical-metric" title="{help_text}"><div class="label">{label}</div><div class="value">{value}</div></div>"""


def format_stage_with_color(stage_string):
    colors = {"Mild": "green", "Moderate": "orange", "Severe": "#D35400", "Critical": "red"}
    for stage, color in colors.items():
        if stage in stage_string: return f"<span style='color: {color};'>{stage_string}</span>"
    return f"<span style='color: #333;'>{stage_string}</span>"


# --- MAIN DASHBOARD DISPLAY (UNCHANGED) ---
def display_dashboard(df, metrics, fft_df, spec_data, corr_matrix, display_info, current_window=None):
    st.title("Advanced Parkinson's Movement Analyzer")
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown("##### Composite Tremor Index")
        fig = go.Figure(
            go.Indicator(mode="gauge+number", value=metrics['composite_index'], number={'valueformat': '.2f'},
                         domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "Severity", 'font': {'size': 16}},
                         gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "dimgray"},
                                'steps': [{'range': [0, st.session_state.stage1_idx], 'color': 'lightgreen'},
                                          {'range': [st.session_state.stage1_idx, st.session_state.stage2_idx],
                                           'color': 'yellow'},
                                          {'range': [st.session_state.stage2_idx, st.session_state.stage3_idx],
                                           'color': 'orange'},
                                          {'range': [st.session_state.stage3_idx, 1], 'color': 'red'}]}))
        fig.update_layout(height=200, margin=dict(t=40, b=10, l=10, r=10), font={'color': "white"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("##### Detailed Metrics for this Window")
        g1, g2, g3 = st.columns(3);
        g4, g5, g6 = st.columns(3)
        g1.markdown(create_metric_box("Tremor Stage", format_stage_with_color(metrics['stage']),
                                      "Overall severity classification."), unsafe_allow_html=True)
        g2.markdown(create_metric_box("RMS Power", f"{metrics['rms_tremor']:.0f}", "Overall tremor intensity."),
                    unsafe_allow_html=True)
        g3.markdown(
            create_metric_box("Stabilizer Effectiveness", f"{metrics['effectiveness']:.1f}%", "Tremor reduction."),
            unsafe_allow_html=True)
        g4.markdown(create_metric_box("Power in 4-8Hz", f"{metrics['power_in_band_ratio']:.1f}%",
                                      "Power in Parkinsonian band."), unsafe_allow_html=True)
        g5.markdown(create_metric_box("Spectral Entropy", f"{metrics['spectral_entropy']:.2f}", "Tremor randomness."),
                    unsafe_allow_html=True)
        g6.markdown(create_metric_box("RMS of Jerk", f"{metrics['rms_jerk'] / 1000:.1f}k", "Movement smoothness."),
                    unsafe_allow_html=True)
    st.divider()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Status", display_info.get('status', 'N/A'));
    s2.metric(display_info.get('id_label', 'ID'), str(display_info.get('id_value', 'N/A')));
    s3.metric("Sample Rate", f"{metrics.get('sampling_freq', 0):.1f} Hz");
    s4.metric(display_info.get('time_label', 'Time'), display_info.get('timestamp', 'N/A'))
    st.write("---")
    tabs = st.tabs(
        ["Movement Overview", "Temporal Frequency", "Movement Dynamics", "Frequency Based", "Clinical Features",
         "Component Details", "Raw Data"])
    with tabs[0]:
        st.subheader(f"Hand Movments vs Stabalizer: {current_window or 'Live Data'}");
        st.info(
            "This chart compares the RAW hand tremor with the movement of the Stabilized spoon . Effective stabilization should show a significantly smaller amplitude for the cyan line (Spoon).")
        fig = go.Figure();
        fig.add_trace(go.Scatter(x=df['time_s'], y=df['total_mag'], mode='lines', name='Raw Hand (Sensor 1)',
                                 line=dict(color='orange')));
        fig.add_trace(
            go.Scatter(x=df['time_s'], y=df['total_mag_stable'], mode='lines', name='Stabilized Spoon (Sensor 2)',
                       line=dict(color='cyan')))
        fig.update_layout(xaxis_title="Time (s)", yaxis_title="Acceleration Magnitude (m/s^2)", template="plotly_dark",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1));
        st.plotly_chart(fig, use_container_width=True)
    with tabs[1]:
        st.subheader("Spectrogram: Tremor Frequency over Time");
        st.info(
            "The spectrogram shows the intensity of different tremor frequencies as they change over time. Bright horizontal bands in the 3–7 Hz range indicate a persistent Parkinsonian tremor.")
        f_spec, t_spec, Sxx = spec_data;
        power_dB = 10 * np.log10(Sxx + 1e-9);
        fig = px.imshow(power_dB, x=t_spec, y=f_spec, aspect='auto',
                        labels=dict(x="Time (s)", y="Frequency (Hz)", color="Power (dB)"),
                        color_continuous_scale='plasma', origin='lower')
        fig.update_yaxes(range=[0, 20]);
        st.plotly_chart(fig, use_container_width=True)
    with tabs[2]:
        st.subheader("Advanced Movement Dynamics");
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("**Poincaré Plot: Tremor Variability**");
            st.info(
                "Plots each acceleration value against the next one. A tight, elongated ellipse indicates a regular tremor. A dispersed, circular cloud suggests random movement.")
            poincare_df = pd.DataFrame({'a_t': df['total_mag'][:-1], 'a_t+1': df['total_mag'][1:]});
            fig = px.scatter(poincare_df, x='a_t', y='a_t+1', opacity=0.6,
                             labels={'a_t': 'Accel. at Time t', 'a_t+1': 'Accel. at Time t+1'}, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("**3D Movement Trajectory**");
            st.info(
                "Visualizes the tremor's path in 3D space. Helps identify if the tremor is directional (linear) or rotational (circular/elliptical).")
            fig = px.scatter_3d(df, x='ax1', y='ay1', z='az1', color='time_s',
                                labels={'ax1': 'X-axis', 'ay1': 'Y-axis', 'az1': 'Z-axis'}, template="plotly_dark",
                                opacity=0.7);
            st.plotly_chart(fig, use_container_width=True)
    with tabs[3]:
        st.subheader("Frequency Spectrum Analysis (FFT)");
        st.info(
            "This breaks down the entire movement signal into its constituent frequencies. A large, sharp peak in the 3-7 Hz range (red band) is a classic signature of Parkinsonian tremor.")
        fig = px.bar(fft_df, x='Frequency (Hz)', y='Power', template="seaborn", log_y=True);
        fig.add_vrect(x0=3, x1=7, fillcolor="red", opacity=0.25, line_width=0,
                      annotation_text="Parkinson's Band (3-7 Hz)")
        fig.update_xaxes(range=[0, 25]);
        st.plotly_chart(fig, use_container_width=True)
    with tabs[4]:
        st.subheader("Clinical Feature Analysis");
        st.info(
            "This section provides a deeper look into specific biomarkers and mathematical features used in clinical research to characterize tremors.")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**Tremor Feature Fingerprint**");
            st.markdown(
                "This radar chart provides a multi-dimensional profile of the tremor. A larger shape indicates a more pronounced tremor across various domains.",
                unsafe_allow_html=True)
            categories = ['Intensity (RMS)', 'Roughness (Jerk)', 'PD-Band Power', 'Irregularity (Entropy)',
                          'Spikiness (Crest)'];
            norm_rms = min(metrics['rms_tremor'] / 5000, 1.0);
            norm_jerk = min(metrics['rms_jerk'] / 150000, 1.0);
            norm_band_power = metrics['band_power_3_7_ratio'] / 100.0;
            norm_entropy = min(metrics['spectral_entropy'] / 7, 1.0);
            norm_crest = min((metrics['crest_factor'] - 1) / 10, 1.0)
            fig = go.Figure();
            fig.add_trace(
                go.Scatterpolar(r=[norm_rms, norm_jerk, norm_band_power, norm_entropy, norm_crest], theta=categories,
                                fill='toself', name='Tremor Profile'))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False,
                              template="plotly_dark", margin=dict(t=40, b=20, l=40, r=40));
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("**Core Biomarkers**");
            st.markdown(create_clinical_metric("Peak Frequency", f"{metrics['peak_freq']:.2f} Hz",
                                               "The single most dominant frequency in the tremor."),
                        unsafe_allow_html=True);
            st.markdown(create_clinical_metric("Power in 3-7Hz Band", f"{metrics['band_power_3_7_ratio']:.1f}%",
                                               "Energy concentrated in the typical Parkinsonian frequency range."),
                        unsafe_allow_html=True);
            st.markdown(create_clinical_metric("Crest Factor", f"{metrics['crest_factor']:.2f}",
                                               "Ratio of peak to RMS power. Higher values indicate spiky tremors."),
                        unsafe_allow_html=True);
            st.markdown(create_clinical_metric("Zero-Crossing Rate", f"{metrics['zcr']:.2f} Hz",
                                               "Indicates signal oscillation frequency."), unsafe_allow_html=True);
            st.markdown(create_clinical_metric("Signal Magnitude Area", f"{metrics['sma']:.1f}",
                                               "Cumulative measure of movement intensity."), unsafe_allow_html=True)
        st.divider();
        col3, col4 = st.columns(2, gap="large")
        with col3:
            st.markdown("**Cross-Axis Correlation**");
            st.info(
                "Shows the correlation between X, Y, and Z axes. High correlation (bright squares) indicates planar tremor; low correlation (dark squares) suggests complex, rotational movement.")
            fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r',
                            range_color=[-1, 1], labels=dict(color="Correlation"));
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            st.markdown("**Per-Axis Movement Variability (Std. Dev.)**");
            st.info(
                "This bar chart shows the standard deviation for each axis, revealing if the tremor is more pronounced in a specific direction.")
            std_df = pd.DataFrame(list(metrics['std_dev_axes'].items()), columns=['Axis', 'Standard Deviation']);
            std_df['Axis'] = std_df['Axis'].map({'ax1': 'X-axis', 'ay1': 'Y-axis', 'az1': 'Z-axis'});
            fig = px.bar(std_df, x='Axis', y='Standard Deviation', color='Axis', template="plotly_dark",
                         text_auto=True);
            st.plotly_chart(fig, use_container_width=True)
    with tabs[5]:
        st.subheader("Per-Axis Time Series Analysis");
        st.info(
            "These charts break down the total movement magnitude into its individual X, Y, and Z components for both the raw hand and the stabilized spoon.")
        c1, c2 = st.columns(2, gap="large")
        with c1: st.markdown("##### Raw Hand Movements (Per Axis)"); st.line_chart(
            df.rename(columns={'ax1': 'X', 'ay1': 'Y', 'az1': 'Z'}).set_index('time_s')[['X', 'Y', 'Z']])
        with c2: st.markdown("##### Stabilized Spoon Movements (Per Axis)"); st.line_chart(
            df.rename(columns={'ax2': 'X', 'ay2': 'Y', 'az2': 'Z'}).set_index('time_s')[['X', 'Y', 'Z']])
    with tabs[6]:
        st.subheader("Raw Analytical Data Table");
        st.info(
            "The raw data points used for the analysis in the current window, including calculated time in seconds, acceleration on each axis, total magnitude, and jerk.")
        display_cols = ['time_s', 'ax1', 'ay1', 'az1', 'ax2', 'ay2', 'az2', 'total_mag', 'jerk'];
        st.dataframe(df[display_cols].style.format("{:.3f}"), use_container_width=True)


# --- NEW PROFESSIONAL REPORT DISPLAY FUNCTION ---
def display_report_page():
    """Renders a professional, print-friendly report page with a white background."""

    report_info = st.session_state.report_data
    if not report_info:
        st.error("No report data found. Returning to dashboard.")
        st.session_state.viewing_report = False
        st.rerun()

    df, metrics, fft_df, _, _ = report_info['analysis']
    patient_details = report_info['details']
    window_str = report_info['window_str']

    # --- Custom CSS for the Report Page (White Theme) ---
    st.markdown("""
    <style>
    /* Hide the default Streamlit header, footer, and toolbar */
    .report-container #stHeader, .report-container .viewerBadge_link__1Srq5, .report-container .styles_toolbar__3_r_5 {
        display: none !important;
    }
    .report-container {
        background-color: #FFFFFF;
        color: #111111;
        padding: 2rem;
        font-family: 'Helvetica', 'Arial', sans-serif;
    }
    .report-container h1, .report-container h2, .report-container h3 {
        color: #003366; /* Dark blue for headings */
    }
    .report-header {
        border-bottom: 2px solid #003366;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
    }
    .report-card {
        border: 1px solid #DDDDDD;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .report-controls {
        text-align: right;
    }
    /* Style Streamlit's table for the report */
    .report-container .stTable {
        font-size: 14px;
    }
    .report-container .stTable > table > tbody > tr > td,
    .report-container .stTable > table > thead > tr > th {
        border: 1px solid #ccc;
    }
    .report-footer {
        text-align: center;
        margin-top: 3rem;
        font-size: 0.8rem;
        color: #888888;
        border-top: 1px solid #DDDDDD;
        padding-top: 1rem;
    }
    /* Styles for printing */
    @media print {
        /* Hide sidebar and report controls when printing */
        .main > div[data-testid="stSidebar"], .report-controls {
            display: none !important;
        }
        /* Ensure the main content uses the full page width */
        section[data-testid="st-container"] {
            padding-left: 0 !important;
            padding-right: 0 !important;
        }
        .report-container {
            padding: 0;
            box-shadow: none;
            border: none;
        }
        .report-card {
            box-shadow: none;
            border: 1px solid #ccc;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # --- HTML for Print Button ---
    print_button_html = """
    <button onclick="window.print()" style="padding: 8px 16px; font-weight: 600; border-radius: 8px; border: 1px solid #003366; background-color: #FFFFFF; color: #003366; cursor: pointer; margin-left: 10px;">
        🖨️ Print Report
    </button>
    """

    # --- Report Layout ---
    with st.container():
        st.markdown('<div class="report-container">', unsafe_allow_html=True)

        # 1. Header
        with st.container():
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(
                    '<div class="report-header"><h1>Parkinson\'s Tremor Analysis Report</h1><h3>Confidential Clinical Data</h3></div>',
                    unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="report-controls">', unsafe_allow_html=True)
                if st.button("⬅️ Back to Dashboard", use_container_width=True):
                    st.session_state.viewing_report = False
                    st.session_state.report_data = None
                    st.rerun()
                st.markdown(print_button_html, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # 2. Patient Information
        with st.container():
            st.markdown("<h3>Patient & Session Information</h3>", unsafe_allow_html=True)
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Patient ID", patient_details.get('patient_id', 'N/A'))
                c2.metric("Recording Date", patient_details.get('timestamp', 'N/A').split(" ")[0])
                c3.metric("Analysis Window", window_str)
                c4.metric("Report Generated", datetime.now().strftime('%Y-%m-%d'))

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Key Findings & Gauge
        st.markdown("<h3>Key Findings</h3>", unsafe_allow_html=True)
        with st.container(border=True):
            col1, col2 = st.columns([1.5, 1])
            with col1:
                st.markdown(f"""
                - **Classification:** The tremor is classified as **{metrics['stage']}**.
                - **Severity Index:** A composite score of **{metrics['composite_index']:.2f}** (out of 1.0) was calculated.
                - **Dominant Frequency:** The tremor shows a peak frequency at **{metrics['peak_freq']:.2f} Hz**, which is within the typical 3-7 Hz band for Parkinson's disease.
                - **Stabilizer Performance:** The hardware stabilizer demonstrated an efficiency of **{metrics['effectiveness']:.1f}%** in reducing tremor amplitude.
                """)
            with col2:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=metrics['composite_index'], number={'valueformat': '.2f'},
                    domain={'x': [0, 1], 'y': [0, 1]}, title={'text': f"Severity Index"},
                    gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "darkgray"}, 'steps': [
                        {'range': [0, 0.3], 'color': '#A5D6A7'}, {'range': [0.3, 0.5], 'color': '#FFF59D'},
                        {'range': [0.5, 0.7], 'color': '#FFCC80'}, {'range': [0.7, 1.0], 'color': '#EF9A9A'}
                    ]}))
                fig_gauge.update_layout(height=250, margin=dict(t=80, b=20), template='plotly_white')
                st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 4. Graphical Analysis
        st.markdown("<h3>Graphical Analysis</h3>", unsafe_allow_html=True)
        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<h5>Time-Series Movement</h5>", unsafe_allow_html=True)
                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(x=df['time_s'], y=df['total_mag'], mode='lines', name='Raw Hand'))
                fig_ts.add_trace(go.Scatter(x=df['time_s'], y=df['total_mag_stable'], mode='lines', name='Stabilized'))
                fig_ts.update_layout(xaxis_title="Time (s)", yaxis_title="Acceleration Magnitude",
                                     template='plotly_white',
                                     legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="left", x=0))
                st.plotly_chart(fig_ts, use_container_width=True)
            with c2:
                st.markdown("<h5>Frequency Spectrum (FFT)</h5>", unsafe_allow_html=True)
                fig_fft = px.bar(fft_df, x='Frequency (Hz)', y='Power', log_y=True)
                fig_fft.add_vrect(x0=3, x1=7, fillcolor="rgba(255,0,0,0.15)", line_width=0, annotation_text="PD Band")
                fig_fft.update_xaxes(range=[0, 25])
                fig_fft.update_layout(template='plotly_white')
                st.plotly_chart(fig_fft, use_container_width=True)

        # 5. Detailed Metrics Table
        st.markdown("<h3>Detailed Clinical Metrics</h3>", unsafe_allow_html=True)
        with st.container(border=True):
            metrics_df = pd.DataFrame({
                "Metric": ["RMS Power", "Peak Frequency", "Power in 3–7 Hz", "RMS Jerk", "Stabilizer Effectiveness",
                           "Spectral Entropy", "Crest Factor"],
                "Value": [f"{metrics['rms_tremor']:.0f}", f"{metrics['peak_freq']:.2f} Hz",
                          f"{metrics['band_power_3_7_ratio']:.1f}%", f"{metrics['rms_jerk'] / 1000:.1f}k",
                          f"{metrics['effectiveness']:.1f}%", f"{metrics['spectral_entropy']:.2f}",
                          f"{metrics['crest_factor']:.2f}"],
                "Description": ["Overall intensity of the tremor.", "The most dominant frequency component.",
                                "Percentage of tremor power in the typical PD range.",
                                "A measure of the movement's smoothness (lower is smoother).",
                                "Reduction in tremor amplitude by the device.",
                                "Measure of the tremor's randomness or unpredictability.",
                                "Ratio of peak to average power, indicating spikiness."]
            })
            st.table(metrics_df.set_index("Metric"))

        st.markdown(
            '<p class="report-footer">This report was automatically generated. For diagnostic purposes, please consult a qualified medical professional.</p>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# --- SIDEBAR & MAIN LOGIC ---
placeholder = st.empty()
analysis_result_for_display = None

with st.sidebar:
    # Hide sidebar controls when viewing the report for a cleaner print view
    if not st.session_state.viewing_report:
        st.title("Dashboard Controls")
        st.session_state.mode = st.radio("Select Mode", ('Live', 'Playback'), horizontal=True,
                                         help="Choose 'Live' to see real-time data or 'Playback' to review saved recordings.")
        st.divider()

        if st.session_state.mode == 'Live':
            st.header("Live Dashboard Settings")
            st.session_state.is_running = st.toggle("Enable Auto-Refresh", value=True,
                                                    disabled=st.session_state.is_recording)
            refresh_interval = st.slider("Update Interval (seconds)", 2, 15, 2, key="live_refresh_interval")
            with st.expander("Continuous Recording", expanded=True):
                st.session_state.patient_id_input = st.text_input("Patient ID", value=st.session_state.patient_id_input,
                                                                  disabled=st.session_state.is_recording)
                if not st.session_state.is_recording:
                    if st.button("Start Recording", use_container_width=True, type="primary"):
                        st.session_state.is_recording = True;
                        st.session_state.recorded_data_buffer = [];
                        st.session_state.last_recorded_id = -1;
                        st.rerun()
                else:
                    st.info(f"🔴 Recording... | {len(st.session_state.recorded_data_buffer)} data points")
                    if st.button("Stop & Save Recording", use_container_width=True):
                        st.session_state.is_recording = False
                        save_recording_to_firebase(st.session_state.patient_id_input,
                                                   f"rec-upto-{st.session_state.last_recorded_id}",
                                                   st.session_state.recorded_data_buffer, FIREBASE_URL, DB_SECRET)
                        st.session_state.recorded_data_buffer = [];
                        st.rerun()
            st.divider()
            st.header("Connection Health")
            total_req = st.session_state.connection_successes + st.session_state.connection_failures
            success_rate = (st.session_state.connection_successes / total_req * 100) if total_req > 0 else 100
            st.metric("Last Fetch", st.session_state.last_conn_status);
            st.metric("Latency", f"{st.session_state.last_latency:.2f} s");
            st.metric("Data Rate", f"{st.session_state.last_data_rate:.1f} KB/s")
            st.progress(success_rate / 100, text=f"Success Rate : {success_rate:.1f}%")

        else:  # Playback Mode
            st.header("Playback Controls");
            st.session_state.is_running = False
            if st.session_state.selected_recording_id is None:
                st.info("Select a recording from the list below to begin playback and analysis.")
                with st.spinner("Loading recordings..."):
                    st.session_state.recordings_list = get_recordings_list(FIREBASE_URL, DB_SECRET)
                if not st.session_state.recordings_list:
                    st.warning("No recordings found.")
                else:
                    options = {rec_id: f"{details['patient_id']} - {details['timestamp']}" for rec_id, details in
                               st.session_state.recordings_list.items()}
                    selection = st.selectbox("Choose a recording to analyze", options=options.keys(),
                                             format_func=lambda rec_id: options[rec_id], index=None,
                                             placeholder="Select a recording...")
                    if selection: st.session_state.selected_recording_id = selection; st.rerun()
            else:
                if st.session_state.full_playback_df is None:
                    with st.spinner("Loading full recording data..."):
                        recording_data = get_specific_recording(st.session_state.selected_recording_id, FIREBASE_URL,
                                                                DB_SECRET)
                        if recording_data and 'data' in recording_data:
                            df, metrics, _, _, _ = perform_advanced_analysis(recording_data['data'])
                            if df is not None:
                                st.session_state.full_playback_df = df;
                                st.session_state.total_duration = metrics.get('duration', 0);
                                st.session_state.current_window_start = 0.0
                            else:
                                st.error(
                                    "Failed to process recording data."); st.session_state.selected_recording_id = None;
                        else:
                            st.error("Failed to load recording data."); st.session_state.selected_recording_id = None;
                    st.rerun()
                if st.session_state.full_playback_df is not None:
                    total_duration = st.session_state.total_duration;
                    start_time = st.session_state.current_window_start;
                    end_time = start_time + 2.0
                    window_df_raw = st.session_state.full_playback_df[
                        (st.session_state.full_playback_df['time_s'] >= start_time) & (
                                    st.session_state.full_playback_df['time_s'] < end_time)]
                    st.session_state.current_window_analysis = perform_advanced_analysis(
                        window_df_raw.to_dict('records'))
                    analysis_result_for_display = st.session_state.current_window_analysis
                    st.subheader("Window Navigation");
                    st.write(f"**Viewing:** `{start_time:.1f}s - {end_time:.1f}s` of `{total_duration:.1f}s` total.")
                    if total_duration and isinstance(total_duration, (int, float)) and total_duration > 2.0:
                        col1, col2 = st.columns(2)
                        if col1.button("Previous Window", use_container_width=True,
                                       disabled=bool(start_time <= 0)): st.session_state.current_window_start = max(0.0,
                                                                                                                    st.session_state.current_window_start - 2.0); st.rerun()
                        if col2.button("Next Window", use_container_width=True, disabled=bool(
                            end_time >= total_duration)): st.session_state.current_window_start = min(
                            total_duration - 2.0, st.session_state.current_window_start + 2.0); st.rerun()
                    st.divider()
                    st.subheader("Generate Report")
                    if st.session_state.current_window_analysis:
                        window_str = f"{start_time:.1f}s - {end_time:.1f}s"
                        if st.button(f"View Report for {window_str}", use_container_width=True, type="primary"):
                            st.session_state.report_data = {"analysis": st.session_state.current_window_analysis,
                                                            "details": st.session_state.recordings_list[
                                                                st.session_state.selected_recording_id],
                                                            "window_str": window_str}
                            st.session_state.viewing_report = True;
                            st.rerun()
                    else:
                        st.info("No data in this window to generate a report.")
                    st.divider()
                    st.subheader("Session Control")
                    if st.button("Stop Playback (Back to List)", use_container_width=True):
                        keys_to_reset = ['selected_recording_id', 'full_playback_df', 'current_window_analysis'];
                        [st.session_state.pop(key, None) for key in keys_to_reset];
                        st.rerun()
                    with st.expander("⚠️ Delete this recording"):
                        st.warning("This action is permanent and cannot be undone.")
                        if st.button("Confirm Deletion", use_container_width=True, type="primary"):
                            if delete_recording_from_firebase(st.session_state.selected_recording_id, FIREBASE_URL,
                                                              DB_SECRET):
                                get_recordings_list.clear();
                                keys_to_reset = ['selected_recording_id', 'full_playback_df',
                                                 'current_window_analysis'];
                                [st.session_state.pop(key, None) for key in keys_to_reset];
                                st.rerun()
        st.divider()
        with st.expander("Analysis & Staging Tuning"):
            st.session_state.w_rms = st.slider("RMS Weight", 0.0, 1.0, 0.4, 0.05);
            st.session_state.w_freq = st.slider("Frequency Weight", 0.0, 1.0, 0.4, 0.05);
            st.session_state.w_jerk = st.slider("Smoothness Weight", 0.0, 1.0, 0.2, 0.05)
            st.session_state.stage1_idx = st.slider("Stage 1/2 Boundary", 0.0, 1.0, 0.3);
            st.session_state.stage2_idx = st.slider("Stage 2/3 Boundary", 0.0, 1.0, 0.5);
            st.session_state.stage3_idx = st.slider("Stage 3/4 Boundary", 0.0, 1.0, 0.7)

# --- MAIN LOGIC WRAPPER ---
if st.session_state.get('viewing_report', False):
    placeholder.empty()
    display_report_page()
else:
    if st.session_state.mode == 'Live':
        st.session_state.current_window_analysis = None
        dataset_id, raw_data, diagnostics = get_live_data_from_firebase(URL=FIREBASE_URL, KEY=DB_SECRET)
        st.session_state.last_latency, st.session_state.last_data_rate = diagnostics['latency'], diagnostics[
            'data_rate']
        if dataset_id is not None:
            if dataset_id != st.session_state.last_seen_id:
                st.session_state.device_status = "🟢 Online";
                st.session_state.last_seen_id = dataset_id;
                st.session_state.last_id_time = time.time()
            elif time.time() - st.session_state.last_id_time > (st.session_state.get('live_refresh_interval', 5) * 4):
                st.session_state.device_status = "🔴 Offline"
            else:
                st.session_state.device_status = "🟡 Moderate"
            if st.session_state.is_recording and raw_data:
                if dataset_id > st.session_state.get('last_recorded_id',
                                                     -1): st.session_state.recorded_data_buffer.extend(
                    raw_data); st.session_state.last_recorded_id = dataset_id
            processed_result = perform_advanced_analysis(raw_data)
            if processed_result:
                df, metrics, fft_df, spec_data, corr_matrix = processed_result
                live_info = {'status': st.session_state.device_status, 'id_label': 'Dataset ID', 'id_value': dataset_id,
                             'time_label': 'Last Update', 'timestamp': datetime.now().strftime('%H:%M:%S')}
                placeholder.empty();
                display_dashboard(df, metrics, fft_df, spec_data, corr_matrix, display_info=live_info)
            else:
                placeholder.warning("Received data is invalid or too small. Waiting for new data...")
        else:
            st.session_state.device_status = "Disconnected"; placeholder.error(
                "Could not retrieve live data. Check connection and secrets. Retrying...")
    else:
        if analysis_result_for_display:
            df, metrics, fft_df, spec_data, corr_matrix = analysis_result_for_display
            start_time = st.session_state.current_window_start;
            window_str = f"{start_time:.1f}s - {start_time + 2.0:.1f}s"
            playback_info = {'status': 'Playback', 'id_label': 'Recording ID',
                             'id_value': st.session_state.selected_recording_id, 'time_label': 'Recorded On',
                             'timestamp': st.session_state.recordings_list[st.session_state.selected_recording_id][
                                 'timestamp']}
            placeholder.empty();
            display_dashboard(df, metrics, fft_df, spec_data, corr_matrix, display_info=playback_info,
                              current_window=window_str)
        elif st.session_state.selected_recording_id is not None and st.session_state.full_playback_df is not None:
            placeholder.warning(f"No valid data in the selected window. Please try another window.")
        elif st.session_state.selected_recording_id is None:
            placeholder.info("Select a recording from the sidebar to begin playback and analysis.")

    # --- Footer for Dashboard ---
    st.markdown(
        """<div style='background-color: #0e1117; color: #4f4f4f; text-align: center; padding: 15px; font-size: 14px; margin-top: 50px; width: 100%; border-top: 1px solid #4f4f4f;'><b>Project:</b> Vibration Analyzed Smart Glove to Aid Parkinson's Patient Hand Tremor with Postural Stability<br><b>Team:</b> 22LE1-035 S.A.P.U.Hemachandra | 22LE2-082 I.H.C.Udayanga | <b>Group:</b> B 07-18<br><b>Supervisor:</b> Mr. Nuwan Attanayake</div>""",
        unsafe_allow_html=True)
    # --- Auto-Refresh Logic ---
    if st.session_state.is_running and st.session_state.mode == 'Live':
        time.sleep(st.session_state.get('live_refresh_interval', 5));
        st.rerun()