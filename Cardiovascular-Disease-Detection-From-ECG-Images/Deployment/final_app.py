import streamlit as st
from Ecg import  ECG

st.set_page_config(page_title="Myocardia", page_icon="💓", layout="wide")

# simple styling with requested color
st.markdown(
  """
  <style>
    :root { --primary: #c38888; }
    .stApp {
      background: linear-gradient(135deg, #ffffff 0%, #fff5f5 45%, #f3dada 100%);
    }
    h1, h2, h3 { color: #c38888; }
    .stButton>button, .stDownloadButton>button {
      background-color: #c38888 !important;
      color: #ffffff !important;
      border: 0 !important;
      border-radius: 10px !important;
      padding: 0.6rem 1rem !important;
      box-shadow: 0 4px 12px rgba(195,136,136,.25);
    }
    .stTabs [data-baseweb="tab-list"] { gap: .25rem; }
    .stTabs [data-baseweb="tab"] {
      background: #fff5f5;
      border-radius: 999px;
      padding: .5rem 1rem;
      color: #2b2b2b;
      border: 1px solid #f0d9d9;
    }
    .stTabs [aria-selected="true"] {
      background: #c38888 !important;
      color: #ffffff !important;
      border-color: #c38888 !important;
    }
    .section-card {
      background: rgba(255,245,245,.9);
      border: 1px solid #f0d9d9;
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 8px 20px rgba(195,136,136,.15);
    }
  </style>
  """,
  unsafe_allow_html=True,
)

st.title("Myocardia")
st.caption("Upload an ECG image – processing will auto-run through to prediction.")

# initialize objects and session state holders
ecg = ECG()
if "uploaded_image" not in st.session_state:
  st.session_state.uploaded_image = None
if "dividing_leads" not in st.session_state:
  st.session_state.dividing_leads = None
if "signals_ready" not in st.session_state:
  st.session_state.signals_ready = False
if "signal_df" not in st.session_state:
  st.session_state.signal_df = None
if "reduced_df" not in st.session_state:
  st.session_state.reduced_df = None
if "prediction" not in st.session_state:
  st.session_state.prediction = None
if "auto_run_complete" not in st.session_state:
  st.session_state.auto_run_complete = False

tab_upload, tab_leads, tab_pre, tab_contours, tab_signals, tab_reduce, tab_predict = st.tabs([
  "Upload",
  "Leads",
  "Preprocess",
  "Contours",
  "Signals",
  "Dimensionality Reduction",
  "Prediction",
])

with tab_upload:
  st.subheader("Upload ECG Image")
  uploaded_file = st.file_uploader("Choose an ECG image (.png/.jpg)")
  if uploaded_file is not None:
    st.session_state.uploaded_image = ecg.getImage(uploaded_file)
    st.image(st.session_state.uploaded_image, width='stretch')
    # reset downstream state on new upload
    st.session_state.dividing_leads = None
    st.session_state.signals_ready = False
    st.session_state.signal_df = None
    st.session_state.reduced_df = None
    st.session_state.prediction = None
    st.session_state.auto_run_complete = False

    # auto-run pipeline with progress
    progress = st.progress(0, text="Starting processing…")
    try:
      progress.progress(10, text="Dividing into leads…")
      st.session_state.dividing_leads = ecg.DividingLeads(st.session_state.uploaded_image)

      progress.progress(30, text="Preprocessing leads…")
      ecg.PreprocessingLeads(st.session_state.dividing_leads)

      progress.progress(50, text="Extracting contours and 1D signals…")
      ecg.SignalExtraction_Scaling(st.session_state.dividing_leads)
      st.session_state.signals_ready = True

      progress.progress(65, text="Combining signals…")
      st.session_state.signal_df = ecg.CombineConvert1Dsignal()

      progress.progress(80, text="Running PCA…")
      st.session_state.reduced_df = ecg.DimensionalReduciton(st.session_state.signal_df)

      progress.progress(95, text="Predicting…")
      st.session_state.prediction = ecg.ModelLoad_predict(st.session_state.reduced_df)
      progress.progress(100, text="Done")
      st.session_state.auto_run_complete = True
      st.success(st.session_state.prediction)
    except Exception as e:
      st.session_state.auto_run_complete = False
      st.error(f"Processing failed: {e}")

with tab_leads:
  st.subheader("Divide into Leads")
  if st.session_state.uploaded_image is None:
    st.info("Upload an image first in the Upload tab.")
  else:
    if st.session_state.dividing_leads is None:
      st.session_state.dividing_leads = ecg.DividingLeads(st.session_state.uploaded_image)
    cols = st.columns(2)
    with cols[0]:
      st.image('Leads_1-12_figure.png', caption="Leads 1-12", width='stretch')
    with cols[1]:
      st.image('Long_Lead_13_figure.png', caption="Lead 13", width='stretch')

with tab_pre:
  st.subheader("Preprocess Leads")
  if st.session_state.dividing_leads is None:
    st.info("Complete the Leads step first.")
  else:
    ecg.PreprocessingLeads(st.session_state.dividing_leads)
    cols = st.columns(2)
    with cols[0]:
      st.image('Preprossed_Leads_1-12_figure.png', caption="Preprocessed Leads 1-12", width='stretch')
    with cols[1]:
      st.image('Preprossed_Leads_13_figure.png', caption="Preprocessed Lead 13", width='stretch')

with tab_contours:
  st.subheader("Extract Contours")
  if st.session_state.dividing_leads is None:
    st.info("Complete the Leads step first.")
  else:
    ecg.SignalExtraction_Scaling(st.session_state.dividing_leads)
    st.image('Contour_Leads_1-12_figure.png', caption="Contours (Leads 1-12)", width='stretch')
    st.session_state.signals_ready = True

with tab_signals:
  st.subheader("Combine 1D Signals")
  if not st.session_state.signals_ready:
    st.info("Run Contours step to extract signals.")
  else:
    st.session_state.signal_df = ecg.CombineConvert1Dsignal()
    st.dataframe(st.session_state.signal_df, width='stretch')

with tab_reduce:
  st.subheader("Dimensionality Reduction (PCA)")
  if st.session_state.signal_df is None:
    st.info("Generate signals first.")
  else:
    st.session_state.reduced_df = ecg.DimensionalReduciton(st.session_state.signal_df)
    st.dataframe(st.session_state.reduced_df, width='stretch')

with tab_predict:
  st.subheader("Prediction")
  if st.session_state.reduced_df is None:
    st.info("Run dimensionality reduction first.")
  else:
    st.session_state.prediction = ecg.ModelLoad_predict(st.session_state.reduced_df)
    st.success(st.session_state.prediction)
