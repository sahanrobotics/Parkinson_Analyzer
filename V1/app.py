#pip install streamlit requests numpy pandas plotly scipy

import streamlit as st
import requests
import numpy as np
import pandas as pd
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
from requests.exceptions import RequestException

# --- Page Configuration and Custom CSS ---
st.set_page_config(
    page_title="Parkinson's Movement Analyzer",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
.metric-box {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 20px 16px;
    text-align: center;
    margin-bottom: 16px;
    height: auto;
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

@media (max-width: 768px) {
    .metric-box {
        padding: 14px 12px;
    }
    .metric-box p {
        font-size: 20px;
    }
    .metric-box h4 {
        font-size: 13px;
    }
}
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
if 'connection_successes' not in st.session_state:
    st.session_state.connection_successes = 0
    st.session_state.connection_failures = 0
    st.session_state.last_latency = 0.0
    st.session_state.last_data_rate = 0.0
    st.session_state.last_conn_status = "Initializing"

# --- Sidebar ---
with st.sidebar:
    st.title("Dashboard Settings")
    if 'is_running' not in st.session_state: st.session_state.is_running = True
    st.session_state.is_running = st.toggle("Enable Auto-Refresh", value=st.session_state.is_running)
    refresh_interval = st.slider("Refresh Interval (seconds)", 2, 15, 5)

    with st.expander("Analysis Thresholds", expanded=True):
        st.session_state.stage1_rms = st.slider("Stage 1/2 Boundary (RMS)", 1500, 3500, 2200)
        st.session_state.stage2_rms = st.slider("Stage 2/3 Boundary (RMS)", 1500, 3500, 2300)
        st.session_state.stage3_rms = st.slider("Stage 3/4 Boundary (RMS)", 1500, 3500, 2400)

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


# --- Data Fetching (as per user's requested structure) ---
def get_data_from_firebase(URL, KEY):
    diagnostics = {'latency': 0, 'data_rate': 0}
    try:
        start_time = time.time()
        id_url = f"{URL}/last_dataset_id.json?auth={KEY}"
        r_id = requests.get(id_url, timeout=3);
        r_id.raise_for_status()
        dataset_id = int(r_id.json())

        data_url = f"{URL}/datasets/{dataset_id}.json?auth={KEY}"
        r_data = requests.get(data_url, timeout=5);
        r_data.raise_for_status()

        latency = time.time() - start_time
        data_size_kb = len(r_data.content) / 1024

        diagnostics['latency'] = latency
        diagnostics['data_rate'] = data_size_kb / latency if latency > 0 else 0

        st.session_state.connection_successes += 1
        st.session_state.last_conn_status = "Connected"
        return dataset_id, r_data.json(), diagnostics

    except (RequestException, ValueError, TypeError):
        st.session_state.connection_failures += 1
        st.session_state.last_conn_status = "Failed"
        return None, None, diagnostics


# --- Analysis Functions ---
def process_data(data):
    if not data or not isinstance(data, list) or len(data) < 10: return None
    df = pd.DataFrame(data)
    df['time_s'] = (df['t'] - df['t'].iloc[0]) / 1000.0
    df[['ax1', 'ay1', 'az1']] = pd.DataFrame(df['A1'].tolist(), index=df.index)
    df[['ax2', 'ay2', 'az2']] = pd.DataFrame(df['A2'].tolist(), index=df.index)
    duration = df['time_s'].iloc[-1]
    freq = len(df) / duration if duration > 0 else 0
    vib1 = np.sqrt(np.mean(df['ax1'] ** 2 + df['ay1'] ** 2 + df['az1'] ** 2))
    total_signal = np.sqrt(df['ax1'] ** 2 + df['ay1'] ** 2 + df['az1'] ** 2)
    df['jerk'] = np.gradient(total_signal, df['time_s'])
    jerk_peaks, _ = find_peaks(np.abs(df['jerk']), height=np.abs(df['jerk']).mean() * 1.5)
    fft_df, dom_freq = perform_fft_analysis(total_signal, freq)
    metrics = {
        "vib1": vib1, "stage": classify_stage(vib1), "freq": freq,
        "effectiveness": ((vib1 - np.sqrt(
            np.mean(df['ax2'] ** 2 + df['ay2'] ** 2 + df['az2'] ** 2))) / vib1) * 100 if vib1 > 0 else 0,
        "dominant_freq": dom_freq, "peak_to_peak": total_signal.max() - total_signal.min(),
        "crest_factor": total_signal.max() / vib1 if vib1 > 0 else 0, "jerk_peak_count": len(jerk_peaks)
    }
    return df, metrics, fft_df


def classify_stage(rms):
    if rms < st.session_state.stage1_rms: return "Stage 0/1 Mild"
    elif rms < st.session_state.stage2_rms: return "Stage 2 Moderate"
    elif rms < st.session_state.stage3_rms: return "Stage 3 Severe"
    else: return "Stage 4 Critical"


def perform_fft_analysis(signal, freq):
    if freq == 0 or len(signal) < 2: return None, None
    N, T = len(signal), 1.0 / freq
    yf, xf = rfft(signal.to_numpy()), rfftfreq(N, T)
    power_spectrum, freq_bins = np.abs(yf)[1:], xf[1:]
    if len(power_spectrum) == 0: return None, None
    return pd.DataFrame({'Frequency (Hz)': freq_bins, 'Power': power_spectrum}), freq_bins[np.argmax(power_spectrum)]


# --- UI & Plotting Helper Functions ---
def create_metric_box(title, value):
    return f"""<div class="metric-box" title="{title}"><p>{value}</p><h4>{title}</h4></div>"""



def format_stage_with_color(stage_string):
    color = "#333"  # Default color
    if "Stage 0/1" in stage_string: color = "green"
    elif "Stage 2" in stage_string: color = "orange"
    elif "Stage 3" in stage_string: color = "#D35400"
    elif "Stage 4" in stage_string: color = "red"
    return f"<span style='color: {color};'>{stage_string}</span>"

st.markdown("""
<style>
/* --- Modern tab styling --- */
[data-baseweb="tab-list"] {
    background-color: #0e1117;
    padding: 6px 10px;
    border-radius: 10px;
    border: 1px solid #333;
    overflow-x: auto;
}

[data-baseweb="tab"] {
    background-color: #1a1d23;
    color: #aaa;
    border-radius: 8px;
    padding: 10px 16px;
    margin-right: 6px;
    font-weight: 500;
    transition: all 0.2s ease-in-out;
}

[data-baseweb="tab"]:hover {
    background-color: #2b2f36;
    color: #ddd;
}

[data-baseweb="tab"][aria-selected="true"] {
    background-color: #4f4f4f;
    color: white;
    box-shadow: 0 2px 6px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)


def display_dashboard(df, metrics, fft_df, dataset_id, device_status, last_update_time):
    st.title("Parkinson's Movement Analyzer")

    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown("##### Tremor Level (RMS)")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=metrics['vib1'],
            gauge={'axis': {'range': [None, 3500]}, 'bar': {'color': "dimgray"},
                   'steps': [{'range': [0, st.session_state.stage1_rms], 'color': 'lightgreen'},
                             {'range': [st.session_state.stage1_rms, st.session_state.stage2_rms], 'color': 'yellow'},
                             {'range': [st.session_state.stage2_rms, st.session_state.stage3_rms], 'color': 'orange'},
                             {'range': [st.session_state.stage3_rms, 3500], 'color': 'red'}]}))
        fig.update_layout(height=180, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("##### Key Indicators")
        g1, g2, g3 = st.columns(3)
        g1.markdown(create_metric_box("Tremor Stage", format_stage_with_color(metrics['stage'])), unsafe_allow_html=True)
        g2.markdown(create_metric_box("Dominant Freq.", f"{metrics['dominant_freq']:.1f} Hz"), unsafe_allow_html=True)
        g3.markdown(create_metric_box("Sample Rate", f"{metrics['freq']:.0f} Hz"), unsafe_allow_html=True)
        g4, g5, g6 = st.columns(3)
        g4.markdown(create_metric_box("Jerk Peaks", f"{metrics['jerk_peak_count']}"), unsafe_allow_html=True)
        g5.markdown(create_metric_box("Crest Factor", f"{metrics['crest_factor']:.2f}"), unsafe_allow_html=True)
        g6.markdown(create_metric_box("Effectiveness", f"{metrics['effectiveness']:.1f}%"), unsafe_allow_html=True)

    st.divider()
    st.subheader("Status")
    info1, info2, info3 = st.columns(3)
    info1.caption(f"**Dataset ID:** `{dataset_id}`")
    info2.caption(f"**Device Status:** {device_status}")
    info3.caption(f"**Last Update:** {last_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
    #st.divider()

    tab_list = ["Overview", "Attenuation & Smoothness", "Frequency", "Directional", "Raw Data"]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_list)

    with tab1:
        c1, c2 = st.columns(2, gap="large"); c1.markdown("##### Raw Hand Movements"); c1.line_chart(df.rename(columns={'ax1': 'X', 'ay1': 'Y', 'az1': 'Z'})[['X', 'Y', 'Z']]); c2.markdown("##### Stabilized Spoon Movements"); c2.line_chart(df.rename(columns={'ax2': 'X', 'ay2': 'Y', 'az2': 'Z'})[['X', 'Y', 'Z']])
    with tab2:
        c1, c2 = st.columns(2, gap="large"); c1.markdown("##### Movement Attenuation (Difference)"); df['diff_X'] = df['ax1'] - df['ax2']; df['diff_Y'] = df['ay1'] - df['ay2']; df['diff_Z'] = df['az1'] - df['az2']; c1.line_chart(df[['diff_X', 'diff_Y', 'diff_Z']]); c2.markdown("##### Movement Smoothness (Jerk)"); c2.line_chart(df.set_index('time_s')['jerk'])
    with tab3:
        c1, c2 = st.columns([3, 2], gap="large"); c1.markdown("##### Frequency Spectrum (FFT)"); fig = px.bar(fft_df, x='Frequency (Hz)', y='Power', template="seaborn"); fig.add_vrect(x0=4, x1=6, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Parkinson's Range"); c1.plotly_chart(fig, use_container_width=True); c2.markdown("##### Axis Power Contribution"); axis_rms = {'X': np.sqrt(np.mean(df['ax1'] ** 2)), 'Y': np.sqrt(np.mean(df['ay1'] ** 2)), 'Z': np.sqrt(np.mean(df['az1'] ** 2))}; fig = go.Figure(data=[go.Pie(labels=list(axis_rms.keys()), values=list(axis_rms.values()), hole=.4)]); fig.update_layout(margin=dict(t=0, b=0, l=0, r=0)); c2.plotly_chart(fig, use_container_width=True)
    with tab4:
        st.markdown("##### Tremor Direction Analysis (XY Plane)"); df_polar = pd.DataFrame({'Magnitude': np.sqrt(df['ax1'] ** 2 + df['ay1'] ** 2), 'Angle': np.arctan2(df['ay1'], df['ax1'])}); fig = px.scatter_polar(df_polar, r="Magnitude", theta="Angle", color="Magnitude", template="plotly_dark", color_continuous_scale=px.colors.sequential.Plasma_r); st.plotly_chart(fig, use_container_width=True)
    with tab5:
        st.markdown("#### Raw Analytical Data"); st.dataframe(df[['time_s', 'ax1', 'ay1', 'az1', 'ax2', 'ay2', 'az2', 'jerk']], use_container_width=True)

# --- Main Application Logic ---
if 'last_seen_id' not in st.session_state: st.session_state.last_seen_id = 0; st.session_state.last_id_time = time.time()
placeholder = st.empty()
#if st.session_state.is_running: placeholder.info("Fetching latest data...")

dataset_id, raw_data, diagnostics = get_data_from_firebase(URL=FIREBASE_URL, KEY=DB_SECRET)
st.session_state.last_latency, st.session_state.last_data_rate = diagnostics['latency'], diagnostics['data_rate']

if dataset_id is not None:
    device_status = "🟡 Stale"
    if dataset_id != st.session_state.last_seen_id:
        device_status = "🟢 Online"
        st.session_state.last_seen_id = dataset_id
        st.session_state.last_id_time = time.time()
    elif time.time() - st.session_state.last_id_time > (refresh_interval * 4):
        device_status = "🔴 Offline"

    processed_result = process_data(raw_data)
    if processed_result:
        df, metrics, fft_df = processed_result
        placeholder.empty()
        display_dashboard(df, metrics, fft_df, dataset_id, device_status, datetime.fromtimestamp(st.session_state.last_id_time))
    else:
        placeholder.warning("No valid data or sample size is too small. Waiting for new data...")
else:
    placeholder.error("Could not retrieve data from Firebase. Check connection and secrets.")

st.markdown("""
    <div style='
        background-color: #0e1117;
        color: #4f4f4f;
        text-align: center;
        padding: 15px;
        font-size: 14px;
        margin-top: 50px;
        width: 100%;
        border-top: 1px solid #4f4f4f;
    '>
        <b>Project Name:</b> Vibration analyzed smart glove to aid Parkinson's patient hand tremor with postural stability<br>
        <b>22LE1-035</b> S.A.P.U.Hemachandra | <b>22LE2-082</b> I.H.C.Udayanga<br>
        <b>Internal Passed</b> | <b>Group number:</b> B 07-18<br>
        <b>Supervisor:</b> Mr. Nuwan Attanayake
    </div>
""", unsafe_allow_html=True)


# --- ADDED THIS FINAL BLOCK FOR RELIABLE AUTO-REFRESH ---
if st.session_state.is_running:
    time.sleep(refresh_interval)
    st.rerun()
else:
    placeholder.warning("Auto-refresh is paused. Enable it in the sidebar to see live data.")



