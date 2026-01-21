import io
import os
import streamlit as st
import pandas as pd
import joblib

from src.data.load_data import get_data_path, load_dataset
from src.cleanup import force_cleanup

try:
    import psutil
except Exception:
    psutil = None


st.set_page_config(page_title="AVM Demo", layout="wide")
st.title("Automatic Valuation Model — Demo & Memory Helper")

st.sidebar.header("Actions")
action = st.sidebar.radio("Choose action:", [
    "Show system memory",
    "Load sample data",
    "Upload CSV",
    "Upload model",
    "Run cleanup",
])


if action == "Show system memory":
    st.header("System / Process Memory")
    if psutil is None:
        st.info("`psutil` not available. Install from requirements to see memory details.")
    else:
        proc = psutil.Process()
        mi = proc.memory_info()
        st.write({"rss_bytes": mi.rss, "vms_bytes": mi.vms})

elif action == "Load sample data":
    st.header("Load sample data")
    path = get_data_path("small-realtor-data.csv", "raw")
    if path.exists():
        df = load_dataset(path)
        st.session_state["df"] = df
        st.dataframe(df.head())
    else:
        st.error(f"Sample file not found: {path}")

elif action == "Upload CSV":
    st.header("Upload CSV for preview / prediction")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.session_state["df"] = df
        st.dataframe(df.head())

elif action == "Upload model":
    st.header("Upload model (joblib / pickle)")
    uploaded = st.file_uploader("Model file", type=["joblib", "pkl", "sav"]) 
    if uploaded is not None:
        data = uploaded.read()
        model = None
        try:
            model = joblib.load(io.BytesIO(data))
        except Exception:
            import pickle
            try:
                model = pickle.loads(data)
            except Exception as e:
                st.error("Could not load model: " + str(e))
        if model is not None:
            st.session_state["model"] = model
            st.success("Model loaded into session state.")

elif action == "Run cleanup":
    st.header("Garbage collector cleanup")
    names = st.text_input("Comma-separated variable names to delete (leave empty for defaults)")
    if st.button("Run cleanup now"):
        names_list = [n.strip() for n in names.split(",") if n.strip()] if names else None
        res = force_cleanup(names_list, globals())
        st.write(res)


st.markdown("---")
# Prediction panel
model = st.session_state.get("model")
df = st.session_state.get("df")
if model is not None and df is not None:
    st.header("Make predictions")
    max_rows = min(100, len(df))
    n = st.number_input("Rows to use for prediction", min_value=1, max_value=max_rows, value=min(5, max_rows))
    if st.button("Predict"):
        try:
            X = df.iloc[:n]
            preds = model.predict(X)
            st.write(pd.DataFrame({"prediction": preds}))
        except Exception as e:
            st.error("Prediction failed: " + str(e))

st.sidebar.markdown("\n---\nRun with: `streamlit run streamlit_app.py`")
