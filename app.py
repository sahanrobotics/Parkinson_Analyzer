# pip install streamlit requests numpy pandas plotly scipy kaleido

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
import base64  # Needed for embedding images in the HTML report

# --- Page Configuration and Custom CSS ---
st.set_page_config(
    page_title="Advanced Parkinson's Movement Analyzer",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
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
        'last_seen_id': 0, 'last_id_time': time.time(), 'device_status': "Initializing",
        'live_dataset_id': None, 'live_raw_data': None,
        'mode': 'Live', 'patient_id_input': 'P001',
        'recordings_list': None, 'selected_recording_id': None,
        'full_playback_df': None, 'current_window_start': 0.0,
        'total_duration': 0.0,
        'current_window_analysis': None,
        'is_recording': False, 'recorded_data_buffer': [], 'last_recorded_id': -1,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# --- DATA FETCHING & SAVING FUNCTIONS (with enhanced error handling) ---
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
        # This part is now safer against malformed data from Firebase
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


# --- ADVANCED ANALYSIS FUNCTION (with enhanced error handling) ---
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
        # This catch-all makes the function robust against unexpected data formats or processing errors
        st.warning(f"Data analysis failed. This can happen with incomplete or corrupt data segments. Error: {e}")
        return None

# --- HELPER & UI FUNCTIONS (Unchanged) ---
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


import numpy as np
import pandas as pd
from datetime import datetime

import numpy as np
import pandas as pd
from datetime import datetime
import re


def generate_html_report(window_analysis, patient_details, window_str):
    """
    Generates a complete, visually rich, and data-dense report with a modern, professional design.
    It combines high-quality SVG graphs with detailed textual analysis and numerical tables,
    remaining 100% reliable with no external dependencies.
    """
    df, metrics, fft_df, _, _ = window_analysis

    # --- PART 1: SVG ICONS AND STYLES ---

    # Simple, modern SVG icons embedded for use in headers.
    svg_icons = {
        "dashboard": '<svg class="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M4 13h6c.55 0 1-.45 1-1V4c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v8c0 .55.45 1 1 1zm0 8h6c.55 0 1-.45 1-1v-4c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v4c0 .55.45 1 1 1zm10 0h6c.55 0 1-.45 1-1v-8c0-.55-.45-1-1-1h-6c-.55 0-1 .45-1 1v8c0 .55.45 1 1 1zM13 4v4c0 .55.45 1 1 1h6c.55 0 1-.45 1-1V4c0-.55-.45-1-1-1h-6c-.55 0-1 .45-1 1z"/></svg>',
        "chart_line": '<svg class="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M3.5 18.49l6-6.01 4 4L22 6.92l-1.41-1.41-7.09 7.97-4-4L2 16.99z"/></svg>',
        "chart_bar": '<svg class="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M5 9.2h3V19H5zM10.6 5h3v14h-3zm5.6 8h3v6h-3z"/></svg>',
        "list_check": '<svg class="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M3 13h2v-2H3v2zm0 4h2v-2H3v2zm0-8h2V7H3v2zm4 4h14v-2H7v2zm0 4h14v-2H7v2zm0-8h14V7H7v2z"/></svg>',
        "table": '<svg class="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M21 3H3c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM8 17H5v-3h3v3zm0-5H5V9h3v3zm0-5H5V4h3v3zm5 10h-3v-3h3v3zm0-5h-3V9h3v3zm0-5h-3V4h3v3zm5 10h-3v-3h3v3zm0-5h-3V9h3v3zm0-5h-3V4h3v3z"/></svg>',
        "zap": '<svg class="icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" /></svg>',
        "activity": '<svg class="icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M22 12h-4l-3 9L9 3l-3 9H2" /></svg>',
        "sparkles": '<svg class="icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM18 13.5l.375 1.5.375-1.5a2.625 2.625 0 00-1.682-1.682L15 11.25l1.5.375a2.625 2.625 0 001.682-1.682L18 8.25l-.375 1.5-.375-1.5a2.625 2.625 0 00-1.682 1.682L13.5 12l1.5.375a2.625 2.625 0 001.682 1.682z" /></svg>'
    }

    # --- SVG HELPER 1: MOVEMENT (TIME-SERIES) CHART ---
    def create_svg_movement_chart(df):
        data_points = df[['time_s', 'total_mag']]
        if len(data_points) > 400:
            step = len(data_points) // 400
            data_points = data_points.iloc[::step]

        w, h, pad_t, pad_b, pad_l, pad_r = 800, 250, 20, 40, 50, 20
        chart_w, chart_h = w - pad_l - pad_r, h - pad_t - pad_b
        max_time = data_points['time_s'].max() or 1
        max_accel = (data_points['total_mag'].max() or 1) * 1.1

        points_list = [
            ((row['time_s'] / max_time) * chart_w + pad_l,
             pad_t + chart_h - (row['total_mag'] / max_accel) * chart_h)
            for _, row in data_points.iterrows()
        ]
        polyline_points = " ".join([f"{p[0]:.2f},{p[1]:.2f}" for p in points_list])

        area_path = "M" + polyline_points
        if points_list:
            area_path += f" L{points_list[-1][0]:.2f},{h - pad_b} L{points_list[0][0]:.2f},{h - pad_b} Z"

        y_grid_lines = ""
        for i in range(5):
            y_pos = pad_t + (chart_h / 4) * i
            val = max_accel * (1 - i / 4)
            y_grid_lines += f'<line x1="{pad_l}" y1="{y_pos}" x2="{pad_l + chart_w}" y2="{y_pos}" class="grid-line" />'
            y_grid_lines += f'<text x="{pad_l - 8}" y="{y_pos + 4}" class="axis-label y-label">{val:.0f}</text>'

        x_labels = ""
        for i in range(5):
            x_pos = pad_l + (chart_w / 4) * i
            val = max_time * (i / 4)
            x_labels += f'<text x="{x_pos}" y="{h - pad_b + 18}" class="axis-label x-label">{val:.1f}s</text>'

        return f"""
        <svg width="100%" viewBox="0 0 {w} {h}" class="chart">
            <defs>
                <linearGradient id="movement-gradient" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0%" stop-color="var(--primary-color)" stop-opacity="0.2"/>
                    <stop offset="100%" stop-color="var(--primary-color)" stop-opacity="0"/>
                </linearGradient>
            </defs>
            {y_grid_lines}
            <path d="{area_path}" class="area-movement" />
            <polyline points="{polyline_points}" class="line-movement" />
            <line x1="{pad_l}" y1="{h - pad_b}" x2="{w - pad_r}" y2="{h - pad_b}" class="axis-line" />
            {x_labels}
            <text x="{w / 2}" y="{h - 5}" class="axis-label title-label">Time</text>
            <text transform="translate(15, {h / 2}) rotate(-90)" class="axis-label title-label">Acceleration</text>
        </svg>
        """

    # --- SVG HELPER 2: FREQUENCY (SPECTRUM) CHART ---
    def create_svg_frequency_chart(fft_df):
        fft_filt = fft_df[(fft_df['Frequency (Hz)'] >= 1) & (fft_df['Frequency (Hz)'] <= 25)].copy()
        if fft_filt.empty: return "<p class='chart-nodata'>No frequency data to display.</p>"

        fft_filt['log_power'] = np.log1p(fft_filt['Power'])
        max_log_power = fft_filt['log_power'].max() or 1

        w, h, pad_t, pad_b, pad_l, pad_r = 800, 250, 20, 40, 50, 20
        chart_w, chart_h = w - pad_l - pad_r, h - pad_t - pad_b
        max_freq = 25

        band_x_start = pad_l + (3 / max_freq) * chart_w
        band_width = ((7 - 3) / max_freq) * chart_w
        pd_band_rect = f'<rect x="{band_x_start}" y="{pad_t}" width="{band_width}" height="{chart_h}" class="pd-band" />'

        bar_width = (chart_w / max_freq) * 0.7
        bars = "".join([
            f'<rect x="{pad_l + (row["Frequency (Hz)"] / max_freq) * chart_w - bar_width / 2:.2f}" '
            f'y="{pad_t + chart_h - (row["log_power"] / max_log_power) * chart_h:.2f}" '
            f'width="{bar_width}" height="{(row["log_power"] / max_log_power) * chart_h:.2f}" class="bar" rx="1"/>'
            for _, row in fft_filt.iterrows()
        ])

        x_labels = ""
        for freq_label in [5, 10, 15, 20, 25]:
            x_pos = pad_l + (freq_label / max_freq) * chart_w
            x_labels += f'<text x="{x_pos}" y="{h - pad_b + 18}" class="axis-label x-label">{freq_label} Hz</text>'

        return f"""
        <svg width="100%" viewBox="0 0 {w} {h}" class="chart">
            {pd_band_rect}
            <line x1="{pad_l}" y1="{h - pad_b}" x2="{w - pad_r}" y2="{h - pad_b}" class="axis-line" />
            {bars}
            {x_labels}
            <text x="{w / 2}" y="{h - 5}" class="axis-label title-label">Frequency</text>
            <text transform="translate(15, {h / 2}) rotate(-90)" class="axis-label title-label">Power (Log Scale)</text>
        </svg>
        <div class="legend"><span class="legend-item"><span class="legend-box pd-band-legend"></span>Parkinson's Band (3-7 Hz)</span></div>
        """

    # --- PART 2: TEXTUAL ANALYSIS AND DATA GENERATION ---

    # --- Stage Parsing and Styling Logic ---
    stage_text = metrics.get('stage', 'Unknown')

    # Use regex to find number and text, e.g., "Stage 3 - Severe"
    match = re.search(r'(Stage\s*\d+)?\s*-?\s*(Mild|Moderate|Severe|Critical)', stage_text, re.IGNORECASE)

    stage_number_html = ""
    stage_name = stage_text  # Fallback

    if match:
        stage_num_part = match.group(1)
        stage_name_part = match.group(2).capitalize()

        stage_name = stage_name_part
        if stage_num_part:
            stage_number_html = f'<span class="stage-number">{stage_num_part}</span>'

    stage_styles = {
        "Mild": {"color": "#198754", "bg": "#1987541A", "variable": "var(--color-mild)"},
        "Moderate": {"color": "#ffc107", "bg": "#ffc1071A", "variable": "var(--color-moderate)"},
        "Severe": {"color": "#fd7e14", "bg": "#fd7e141A", "variable": "var(--color-severe)"},
        "Critical": {"color": "#dc3545", "bg": "#dc35451A", "variable": "var(--color-critical)"},
    }
    default_style = {"color": "#6c757d", "bg": "#6c757d1A", "variable": "var(--text-secondary)"}
    selected_style = stage_styles.get(stage_name, default_style)

    prediction_color_var = selected_style["variable"]
    prediction_bg_color = selected_style["bg"]

    # --- Advice Logic ---
    if "Mild" in stage_name:
        prediction_advice = "The tremor is minimal. Continued monitoring for any changes is advised."
    elif "Moderate" in stage_name:
        prediction_advice = "The tremor is noticeable and may cause difficulty. Therapeutic intervention could be beneficial."
    elif "Severe" in stage_name:
        prediction_advice = "The tremor is prominent and likely interferes with daily living. Intervention is highly recommended."
    else:  # Critical or Unknown
        prediction_advice = "The data indicates a critical level of motor symptoms requiring prompt review of the current management plan."

    # --- Detailed Observations Logic ---
    stabilized_rms = np.sqrt(np.mean(df['total_mag_stable'] ** 2)) if 'total_mag_stable' in df else 0

    # --- Full Data Table Logic ---
    metric_table_rows = "".join([
        f"<tr><td>{label}</td><td>{val}</td></tr>" for label, val in [
            ("Composite Severity Index", f"{metrics['composite_index']:.3f}"),
            ("Peak Tremor Frequency", f"{metrics['peak_freq']:.2f} Hz"),
            ("Power in Parkinson's Band (3-7Hz)", f"{metrics['band_power_3_7_ratio']:.1f} %"),
            ("Overall Tremor Intensity (RMS)", f"{metrics['rms_tremor']:.0f}"),
            ("Stabilizer Effectiveness", f"{metrics['effectiveness']:.1f} %"),
            ("Movement Jerkiness (RMS Jerk)", f"{metrics['rms_jerk'] / 1000:.1f} k"),
            ("Movement Randomness (Entropy)", f"{metrics['spectral_entropy']:.2f}"),
            ("Crest Factor", f"{metrics['crest_factor']:.2f}"),
            ("Signal Magnitude Area", f"{metrics['sma']:.1f}"),
        ]
    ])

    # --- PART 3: HTML ASSEMBLY ---
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    movement_chart_svg = create_svg_movement_chart(df)
    frequency_chart_svg = create_svg_frequency_chart(fft_df)

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Movement Report: {patient_details.get('patient_id', 'N/A')}</title>
<style>
  :root {{
    --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', Roboto, Helvetica, Arial, sans-serif;
    --font-mono: 'Menlo', 'Consolas', 'SFMono-Regular', 'source-code-pro', monospace;
    --bg-color: #f8f9fa;
    --card-bg-color: #ffffff;
    --text-primary: #212529;
    --text-secondary: #6c757d;
    --border-color: #e9ecef;
    --primary-color: #0d6efd;
    --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    --border-radius: 0.75rem;
    --color-mild: #198754;
    --color-moderate: #ffc107;
    --color-severe: #fd7e14;
    --color-critical: #dc3545;
    --prediction-color: {prediction_color_var};
  }}
  body {{
    font-family: var(--font-sans); margin: 0; padding: 2rem 1rem;
    background-color: var(--bg-color); color: var(--text-primary);
    -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
  }}
  .container {{
    max-width: 900px; margin: 0 auto; background-color: var(--card-bg-color);
    box-shadow: var(--shadow); border-radius: var(--border-radius); overflow: hidden;
  }}
  .header {{
    padding: 2rem 2.5rem; border-bottom: 1px solid var(--border-color);
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;
  }}
  .header h1 {{ margin: 0; font-size: 1.75rem; font-weight: 700; color: var(--text-primary); }}
  .header-meta {{ text-align: right; font-size: 0.9rem; color: var(--text-secondary); line-height: 1.5; }}
  .content {{ padding: 2.5rem; display: grid; gap: 2.5rem; }}
  .card-header {{
    display: flex; align-items: center; gap: 0.75rem; padding-bottom: 1rem;
    margin-bottom: 1.5rem; border-bottom: 1px solid var(--border-color);
  }}
  .card-header .icon {{ flex-shrink: 0; width: 24px; height: 24px; color: var(--primary-color); }}
  .card-header h2 {{ font-size: 1.25rem; font-weight: 600; margin: 0; color: var(--text-primary); }}

  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; }}
  .metric-card {{ background-color: #f8f9fa; padding: 1.25rem; border-radius: 0.5rem; border: 1px solid var(--border-color); transition: transform 0.2s ease, box-shadow 0.2s ease; }}
  .metric-card:hover {{ transform: translateY(-3px); box-shadow: var(--shadow); }}
  .metric-card-title {{ font-size: 0.9rem; color: var(--text-secondary); margin: 0 0 0.5rem 0; }}
  .metric-card-value {{ font-size: 1.75rem; font-weight: 700; margin: 0; line-height: 1.2; }}
  .metric-card-unit {{ font-size: 1rem; font-weight: 500; color: var(--text-secondary); margin-left: 0.25rem; }}

  .severity-card {{ background-color: {prediction_bg_color}; border-left: 5px solid var(--prediction-color); }}
  .severity-card .metric-card-title {{ color: var(--prediction-color); font-weight: 500; }}
  .severity-card .metric-card-value {{ color: var(--prediction-color); font-size: 2rem; }}
  .stage-number {{ display: block; font-size: 1rem; font-weight: 500; opacity: 0.8; margin-bottom: 0.25rem; }}

  .advice-card {{ padding: 1.25rem; border-radius: 0.5rem; background-color: #e9ecef40; border-left: 4px solid var(--text-secondary); font-size: 0.95rem; line-height: 1.6; }}
  .advice-card p {{ margin: 0; color: var(--text-primary); }}

  .observation-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; }}
  .observation-card {{ background: #fff; border: 1px solid var(--border-color); border-radius: 0.5rem; padding: 1.25rem; }}
  .observation-card-header {{ display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem; }}
  .observation-card-icon {{ flex-shrink: 0; width: 24px; height: 24px; color: var(--primary-color); }}
  .observation-card-title {{ font-size: 1rem; font-weight: 600; margin: 0; }}
  .observation-card-body {{ font-size: 0.9rem; line-height: 1.6; color: var(--text-secondary); }}
  .observation-card-body strong {{ color: var(--text-primary); font-weight: 600; }}

  .data-table {{ width: 100%; border-collapse: collapse; font-size: 0.95rem; }}
  .data-table tr {{ border-bottom: 1px solid var(--border-color); }}
  .data-table tr:last-child {{ border-bottom: none; }}
  .data-table td {{ padding: 1rem 0.5rem; vertical-align: middle; }}
  .data-table td:first-child {{ color: var(--text-secondary); }}
  .data-table td:last-child {{ font-family: var(--font-mono); font-weight: 600; text-align: right; color: var(--text-primary); }}

  .chart-container {{ border: 1px solid var(--border-color); border-radius: 0.5rem; padding: 1rem; background-color: var(--card-bg-color); }}
  .chart-nodata {{ padding: 3rem 1rem; text-align: center; color: var(--text-secondary); }}
  .chart {{ background-color: var(--card-bg-color); }}
  .grid-line {{ stroke: #eef2f6; stroke-width: 1; }}
  .axis-line {{ stroke: #adb5bd; stroke-width: 1; }}
  .axis-label {{ font-size: 11px; fill: var(--text-secondary); }}
  .y-label, .x-label {{ text-anchor: middle; }}
  .y-label {{ text-anchor: end; }}
  .title-label {{ font-weight: 500; fill: var(--text-primary); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .line-movement {{ fill: none; stroke: var(--primary-color); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
  .area-movement {{ fill: url(#movement-gradient); }}
  .bar {{ fill: var(--primary-color); }}
  .pd-band {{ fill: var(--color-critical); opacity: 0.1; }}
  .legend {{ padding-top: 10px; text-align: center; font-size: 12px; color: var(--text-secondary); }}
  .legend-item {{ display: inline-flex; align-items: center; margin: 0 10px; }}
  .legend-box {{ width: 14px; height: 14px; margin-right: 5px; border-radius: 3px; }}
  .pd-band-legend {{ background-color: rgba(220, 53, 69, 0.3); }}

  .footer {{ text-align: center; color: var(--text-secondary); padding: 2.5rem; background-color: var(--bg-color); }}
  .footer .timestamp {{ font-size: 0.8rem; margin-bottom: 1.5rem; }}
  .project-footer {{ border-top: 1px solid var(--border-color); padding-top: 1.5rem; font-size: 0.85rem; line-height: 1.6; }}
  .project-footer p {{ margin: 0.25rem 0; }}
  .project-footer .title {{ font-weight: 600; color: var(--text-primary); }}
</style>
</head>
<body>
  <div class="container">
    <header class="header">
        <h1>Movement Analysis Report</h1>
        <div class="header-meta">
            <strong>Patient:</strong> {patient_details.get('patient_id', 'N/A')}<br>
            <strong>Date:</strong> {patient_details.get('timestamp', 'N/A')} | <strong>Window:</strong> {window_str}
        </div>
    </header>
    <main class="content">
        <section class="card">
            <div class="card-header">{svg_icons["dashboard"]}<h2>Analysis Summary</h2></div>
            <div class="metrics-grid">
                <div class="metric-card severity-card">
                    <p class="metric-card-title">Predicted Severity</p>
                    <p class="metric-card-value">{stage_number_html}{stage_name}</p>
                </div>
                <div class="metric-card">
                    <p class="metric-card-title">Composite Index</p>
                    <p class="metric-card-value">{metrics['composite_index']:.3f}</p>
                </div>
                <div class="metric-card">
                    <p class="metric-card-title">Peak Frequency</p>
                    <p class="metric-card-value">{metrics['peak_freq']:.2f}<span class="metric-card-unit">Hz</span></p>
                </div>
                <div class="metric-card">
                    <p class="metric-card-title">Stabilizer Effectiveness</p>
                    <p class="metric-card-value">{metrics['effectiveness']:.1f}<span class="metric-card-unit">%</span></p>
                </div>
            </div>
            <div style="margin-top: 1.5rem;" class="advice-card">
                <p><strong>Clinical Advice:</strong> {prediction_advice}</p>
            </div>
        </section>

        <section class="card">
            <div class="card-header">{svg_icons["list_check"]}<h2>Detailed Observations</h2></div>
            <div class="observation-grid">
                <div class="observation-card">
                    <div class="observation-card-header">
                        <div class="observation-card-icon">{svg_icons["zap"]}</div>
                        <h3 class="observation-card-title">Intensity & Effectiveness</h3>
                    </div>
                    <div class="observation-card-body">
                        Raw tremor power was <strong>{metrics['rms_tremor']:.0f} RMS</strong>. The device reduced this to 
                        <strong>{stabilized_rms:.0f} RMS</strong>, showing an effectiveness of <strong>{metrics['effectiveness']:.1f}%</strong>.
                    </div>
                </div>
                <div class="observation-card">
                    <div class="observation-card-header">
                        <div class="observation-card-icon">{svg_icons["activity"]}</div>
                        <h3 class="observation-card-title">Frequency Profile</h3>
                    </div>
                    <div class="observation-card-body">
                        A dominant tremor was found at <strong>{metrics['peak_freq']:.2f} Hz</strong>, with <strong>{metrics['band_power_3_7_ratio']:.1f}%</strong> of energy 
                        in the 3-7 Hz Parkinsonian band.
                    </div>
                </div>
                <div class="observation-card">
                    <div class="observation-card-header">
                        <div class="observation-card-icon">{svg_icons["sparkles"]}</div>
                        <h3 class="observation-card-title">Movement Quality</h3>
                    </div>
                    <div class="observation-card-body">
                        Movement smoothness (jerkiness) was <strong>{metrics['rms_jerk'] / 1000:.1f}k</strong>. Entropy of <strong>{metrics['spectral_entropy']:.2f}</strong> suggests a 
                        {'highly regular tremor.' if metrics['spectral_entropy'] < 3.5 else 'somewhat irregular tremor.'}
                    </div>
                </div>
            </div>
        </section>

        <section class="card">
            <div class="card-header">{svg_icons["chart_line"]}<h2>Hand Movement Analysis</h2></div>
            <div class="chart-container">{movement_chart_svg}</div>
        </section>

        <section class="card">
            <div class="card-header">{svg_icons["chart_bar"]}<h2>Frequency Spectrum</h2></div>
            <div class="chart-container">{frequency_chart_svg}</div>
        </section>

        <section class="card">
            <div class="card-header">{svg_icons["table"]}<h2>All Numerical Data</h2></div>
            <table class="data-table">{metric_table_rows}</table>
        </section>
    </main>
    <footer class="footer">
      <p class="timestamp">Report generated on {now}. This is a quantitative report for clinical review and research purposes.</p>
      <div class="project-footer">
          <p class="title">Project: Vibration Analyzed Smart Glove to Aid Parkinson's Patient Hand Tremor with Postural Stability</p>
          <p><strong>Team:</strong> 22LE1-035 S.A.P.U.Hemachandra | 22LE2-082 I.H.C.Udayanga | Group: B 07-18</p>
          <p><strong>Supervisor:</strong> Mr. Nuwan Attanayake</p>
      </div>
    </footer>
  </div>
</body>
</html>
"""
    return html




# --- MAIN DASHBOARD DISPLAY (Unchanged) ---
def display_dashboard(df, metrics, fft_df, spec_data, corr_matrix, display_info, current_window=None):
    st.title("Advanced Parkinson's Movement Analyzer")

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
    s1.metric("Status", display_info.get('status', 'N/A'))
    s2.metric(display_info.get('id_label', 'ID'), str(display_info.get('id_value', 'N/A')))
    s3.metric("Sample Rate", f"{metrics.get('sampling_freq', 0):.1f} Hz")
    s4.metric(display_info.get('time_label', 'Time'), display_info.get('timestamp', 'N/A'))
    st.write("---")

    tabs = st.tabs(
        ["Movement Overview", "Temporal Frequency", "Movement Dynamics", "Frequency Based", "Clinical Features",
         "Component Details", "Raw Data"])

    with tabs[0]:
        st.subheader(f"Hand Movments vs Stabalizer: {current_window or 'Live Data'}")
        st.info(
            "This chart compares the RAW hand tremor with the movement of the Stabilized spoon . Effective stabilization should show a significantly smaller amplitude for the cyan line (Spoon).")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['time_s'], y=df['total_mag'], mode='lines', name='Raw Hand (Sensor 1)',
                                 line=dict(color='orange')))
        fig.add_trace(
            go.Scatter(x=df['time_s'], y=df['total_mag_stable'], mode='lines', name='Stabilized Spoon (Sensor 2)',
                       line=dict(color='cyan')))
        fig.update_layout(xaxis_title="Time (s)", yaxis_title="Acceleration Magnitude (m/s^2)", template="plotly_dark",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)
    with tabs[1]:
        st.subheader("Spectrogram: Tremor Frequency over Time")
        st.info(
            "The spectrogram shows the intensity of different tremor frequencies as they change over time. Bright horizontal bands in the 3–7 Hz range indicate a persistent Parkinsonian tremor.")
        f_spec, t_spec, Sxx = spec_data
        power_dB = 10 * np.log10(Sxx + 1e-9)
        fig = px.imshow(power_dB, x=t_spec, y=f_spec, aspect='auto',
                        labels=dict(x="Time (s)", y="Frequency (Hz)", color="Power (dB)"),
                        color_continuous_scale='plasma', origin='lower')
        fig.update_yaxes(range=[0, 20]);
        st.plotly_chart(fig, use_container_width=True)
    with tabs[2]:
        st.subheader("Advanced Movement Dynamics")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("**Poincaré Plot: Tremor Variability**")
            st.info(
                "Plots each acceleration value against the next one. A tight, elongated ellipse indicates a regular tremor. A dispersed, circular cloud suggests random movement.")
            poincare_df = pd.DataFrame({'a_t': df['total_mag'][:-1], 'a_t+1': df['total_mag'][1:]})
            fig = px.scatter(poincare_df, x='a_t', y='a_t+1', opacity=0.6,
                             labels={'a_t': 'Accel. at Time t', 'a_t+1': 'Accel. at Time t+1'}, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("**3D Movement Trajectory**")
            st.info(
                "Visualizes the tremor's path in 3D space. Helps identify if the tremor is directional (linear) or rotational (circular/elliptical).")
            fig = px.scatter_3d(df, x='ax1', y='ay1', z='az1', color='time_s',
                                labels={'ax1': 'X-axis', 'ay1': 'Y-axis', 'az1': 'Z-axis'}, template="plotly_dark",
                                opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)
    with tabs[3]:
        st.subheader("Frequency Spectrum Analysis (FFT)")
        st.info(
            "This breaks down the entire movement signal into its constituent frequencies. A large, sharp peak in the 3-7 Hz range (red band) is a classic signature of Parkinsonian tremor.")
        fig = px.bar(fft_df, x='Frequency (Hz)', y='Power', template="seaborn", log_y=True)
        fig.add_vrect(x0=3, x1=7, fillcolor="red", opacity=0.25, line_width=0,
                      annotation_text="Parkinson's Band (3-7 Hz)")
        fig.update_xaxes(range=[0, 25]);
        st.plotly_chart(fig, use_container_width=True)
    with tabs[4]:
        st.subheader("Clinical Feature Analysis")
        st.info(
            "This section provides a deeper look into specific biomarkers and mathematical features used in clinical research to characterize tremors.")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**Tremor Feature Fingerprint**")
            st.markdown(
                "This radar chart provides a multi-dimensional profile of the tremor. A larger shape indicates a more pronounced tremor across various domains.",
                unsafe_allow_html=True)
            categories = ['Intensity (RMS)', 'Roughness (Jerk)', 'PD-Band Power', 'Irregularity (Entropy)',
                          'Spikiness (Crest)']
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
            st.markdown("**Core Biomarkers**")
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
        st.divider()
        col3, col4 = st.columns(2, gap="large")
        with col3:
            st.markdown("**Cross-Axis Correlation**")
            st.info(
                "Shows the correlation between X, Y, and Z axes. High correlation (bright squares) indicates planar tremor; low correlation (dark squares) suggests complex, rotational movement.")
            fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r',
                            range_color=[-1, 1], labels=dict(color="Correlation"))
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            st.markdown("**Per-Axis Movement Variability (Std. Dev.)**")
            st.info(
                "This bar chart shows the standard deviation for each axis, revealing if the tremor is more pronounced in a specific direction.")
            std_df = pd.DataFrame(list(metrics['std_dev_axes'].items()), columns=['Axis', 'Standard Deviation'])
            std_df['Axis'] = std_df['Axis'].map({'ax1': 'X-axis', 'ay1': 'Y-axis', 'az1': 'Z-axis'})
            fig = px.bar(std_df, x='Axis', y='Standard Deviation', color='Axis', template="plotly_dark", text_auto=True)
            st.plotly_chart(fig, use_container_width=True)
    with tabs[5]:
        st.subheader("Per-Axis Time Series Analysis")
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


# --- SIDEBAR & MAIN LOGIC (with button emojis removed) ---
placeholder = st.empty()
analysis_result_for_display = None

with st.sidebar:
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
        success_rate = 100-((st.session_state.connection_successes / total_req * 100) if total_req > 0 else 100)
        st.metric("Last Fetch", st.session_state.last_conn_status)
        st.metric("Latency", f"{st.session_state.last_latency:.2f} s")
        st.metric("Data Rate", f"{st.session_state.last_data_rate:.1f} KB/s")
        st.progress(int(success_rate), text=f"Packet Loss : {success_rate:.1f}%")

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
                if selection:
                    st.session_state.selected_recording_id = selection;
                    st.rerun()
        else:  # A recording is selected
            if st.session_state.full_playback_df is None:
                with st.spinner("Loading full recording data..."):
                    recording_data = get_specific_recording(st.session_state.selected_recording_id, FIREBASE_URL,
                                                            DB_SECRET)
                    if recording_data and 'data' in recording_data:
                        df, metrics, _, _, _ = perform_advanced_analysis(recording_data['data'])
                        if df is not None:
                            st.session_state.full_playback_df = df
                            st.session_state.total_duration = metrics.get('duration', 0);
                            st.session_state.current_window_start = 0.0
                        else:
                            st.error("Failed to process recording data.");
                            st.session_state.selected_recording_id = None;
                    else:
                        st.error("Failed to load recording data.");
                        st.session_state.selected_recording_id = None;
                st.rerun()

            if st.session_state.full_playback_df is not None:
                total_duration = st.session_state.total_duration;
                start_time = st.session_state.current_window_start;
                end_time = start_time + 2.0
                window_df_raw = st.session_state.full_playback_df[
                    (st.session_state.full_playback_df['time_s'] >= start_time) & (
                                st.session_state.full_playback_df['time_s'] < end_time)]
                window_data_as_list = window_df_raw.to_dict('records')
                st.session_state.current_window_analysis = perform_advanced_analysis(window_data_as_list)
                analysis_result_for_display = st.session_state.current_window_analysis

                st.subheader("Window Navigation")
                st.write(f"**Viewing:** `{start_time:.1f}s - {end_time:.1f}s` of `{total_duration:.1f}s` total.")
                if total_duration and isinstance(total_duration, (int, float)) and total_duration > 2.0:
                    col1, col2 = st.columns(2)

                    prev_disabled = bool(start_time <= 0)
                    next_disabled = bool(end_time >= total_duration)

                    if col1.button("Previous Window", use_container_width=True, disabled=prev_disabled):
                        st.session_state.current_window_start = max(0.0, st.session_state.current_window_start - 2.0)
                        st.rerun()

                    if col2.button("Next Window", use_container_width=True, disabled=next_disabled):
                        st.session_state.current_window_start = min(total_duration - 2.0,
                                                                    st.session_state.current_window_start + 2.0)
                        st.rerun()

                st.divider()
                st.subheader("Generate Report")
                if st.session_state.current_window_analysis:
                    window_str = f"{start_time:.1f}s - {end_time:.1f}s"
                    html_report = generate_html_report(st.session_state.current_window_analysis,
                                                       st.session_state.recordings_list[
                                                           st.session_state.selected_recording_id], window_str)
                    st.download_button(
                        label=f"Download Report for {window_str}",
                        data=html_report,
                        file_name=f"Report_{st.session_state.recordings_list[st.session_state.selected_recording_id].get('patient_id', 'NA')}_Window_{window_str.replace('s - ', 'to').replace('s', '')}.html",
                        mime="text/html", use_container_width=True
                    )
                else:
                    st.info("No data in this window to generate a report.")

                st.divider()
                st.subheader("Session Control")
                if st.button("Stop Playback (Back to List)", use_container_width=True):
                    keys_to_reset = ['selected_recording_id', 'full_playback_df', 'current_window_analysis']
                    for key in keys_to_reset: st.session_state[key] = None
                    st.rerun()
                with st.expander("⚠️ Delete this recording"):
                    st.warning("This action is permanent and cannot be undone.")
                    if st.button("Confirm Deletion", use_container_width=True, type="primary"):
                        if delete_recording_from_firebase(st.session_state.selected_recording_id, FIREBASE_URL,
                                                          DB_SECRET):
                            get_recordings_list.clear();
                            keys_to_reset = ['selected_recording_id', 'full_playback_df', 'current_window_analysis']
                            for key in keys_to_reset: st.session_state[key] = None
                            st.rerun()

    st.divider()
    with st.expander("Analysis & Staging Tuning"):
        st.session_state.w_rms = st.slider("RMS Weight", 0.0, 1.0, 0.4, 0.05);
        st.session_state.w_freq = st.slider("Frequency Weight", 0.0, 1.0, 0.4, 0.05);
        st.session_state.w_jerk = st.slider("Smoothness Weight", 0.0, 1.0, 0.2, 0.05)
        st.session_state.stage1_idx = st.slider("Stage 1/2 Boundary", 0.0, 1.0, 0.3);
        st.session_state.stage2_idx = st.slider("Stage 2/3 Boundary", 0.0, 1.0, 0.5);
        st.session_state.stage3_idx = st.slider("Stage 3/4 Boundary", 0.0, 1.0, 0.7)

# --- MAIN APPLICATION LOGIC ---
if st.session_state.mode == 'Live':
    st.session_state.current_window_analysis = None
    dataset_id, raw_data, diagnostics = get_live_data_from_firebase(URL=FIREBASE_URL, KEY=DB_SECRET)
    st.session_state.last_latency, st.session_state.last_data_rate = diagnostics['latency'], diagnostics['data_rate']

    if dataset_id is not None:
        if dataset_id != st.session_state.last_seen_id:
            st.session_state.device_status = "🟢 Online"
            st.session_state.last_seen_id = dataset_id
            st.session_state.last_id_time = time.time()
        elif time.time() - st.session_state.last_id_time > (st.session_state.get('live_refresh_interval', 5) * 4):
            st.session_state.device_status = "🔴 Offline"
        else:
            st.session_state.device_status = "🟡 Moderate"

        if st.session_state.is_recording and raw_data:
            # Check to avoid duplicating data from the same dataset_id
            if dataset_id > st.session_state.get('last_recorded_id', -1):
                st.session_state.recorded_data_buffer.extend(raw_data)
                st.session_state.last_recorded_id = dataset_id

        processed_result = perform_advanced_analysis(raw_data)
        if processed_result:
            df, metrics, fft_df, spec_data, corr_matrix = processed_result
            live_info = {
                'status': st.session_state.device_status,
                'id_label': 'Dataset ID',
                'id_value': dataset_id,
                'time_label': 'Last Update',
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
            placeholder.empty();
            display_dashboard(df, metrics, fft_df, spec_data, corr_matrix, display_info=live_info)
        else:
            placeholder.warning("Received data is invalid or too small. Waiting for new data...")
    else:
        st.session_state.device_status = "Disconnected"
        placeholder.error("Could not retrieve live data. Check connection and secrets. Retrying...")

else:  # Playback mode
    if analysis_result_for_display:
        df, metrics, fft_df, spec_data, corr_matrix = analysis_result_for_display
        start_time = st.session_state.current_window_start
        window_str = f"{start_time:.1f}s - {start_time + 2.0:.1f}s"
        playback_info = {
            'status': 'Playback',
            'id_label': 'Recording ID',
            'id_value': st.session_state.selected_recording_id,
            'time_label': 'Recorded On',
            'timestamp': st.session_state.recordings_list[st.session_state.selected_recording_id]['timestamp']
        }
        placeholder.empty()
        display_dashboard(df, metrics, fft_df, spec_data, corr_matrix, display_info=playback_info,
                          current_window=window_str)
    elif st.session_state.selected_recording_id is not None and st.session_state.full_playback_df is not None:
        placeholder.warning(f"No valid data in the selected window. Please try another window.");
    elif st.session_state.selected_recording_id is None:
        placeholder.info("Select a recording from the sidebar to begin playback and analysis.")
    else:
        pass


#Bug Fix
    # --- Footer ---
st.markdown("""
    <div style='background-color: #0e1117; color: #4f4f4f; text-align: center; padding: 15px; font-size: 14px; margin-top: 50px; width: 100%; border-top: 1px solid #4f4f4f;'>
        <b>Project:</b> Vibration Analyzed Smart Glove to Aid Parkinson's Patient Hand Tremor with Postural Stability<br>
        <b>Team:</b> 22LE1-035 S.A.P.U.Hemachandra | 22LE2-082 I.H.C.Udayanga | <b>Group:</b> B 07-18<br>
        <b>Supervisor:</b> Mr. Nuwan Attanayake
    </div>
""", unsafe_allow_html=True)




# --- Auto-Refresh Logic ---
if st.session_state.is_running and st.session_state.mode == 'Live':
    time.sleep(st.session_state.get('live_refresh_interval', 5))
    st.rerun()