import streamlit as st 
import joblib
import pandas as pd
import os from pathlib
import Path
import sys
import plotly.express as pximport sklearn if 'sklearn.compose._column_transformer' in sys.modules: mod = sys.modules['sklearn.compose._column_transformer'] if not hasattr(mod, '_RemainderColsList'): class _RemainderColsList(list): pass mod._RemainderColsList = _RemainderColsList

st.set_page_config( page_title="Municipal Protest Risk Dashboard", page_icon="⚠️", layout="wide" )

--- Constants ---

PROVINCES = [ 'Eastern Cape', 'Free State', 'Gauteng', 'KwaZulu-Natal', 'Limpopo', 'Mpumalanga', 'North West', 'Northern Cape', 'Western Cape' ]

--- Load Model ---

@st.cache_resource def load_model(): try: current_dir = Path(file).parent model_path = current_dir / "protest_risk_model.pkl"

if not model_path.exists():
        st.error(f"❌ Model file not found at: {model_path}")
        return None

    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully!")
    return model
except Exception as e:
    st.error(f"Model loading failed: {str(e)}")
    return None

--- Input Validation ---

def validate_inputs(df): if 'Province name' in df.columns: df['Province name'] = df['Province name'].str.strip().str.title() df['Province name'] = df['Province name'].apply( lambda x: x if x in PROVINCES else 'Unknown' ) num_cols = df.select_dtypes(include=['number']).columns df[num_cols] = df[num_cols].clip(lower=0) return df

--- Feature Importance ---

def get_top_features(model, feature_names, top_n=5): if hasattr(model, "feature_importances_"): importances = model.feature_importances_ importance_df = pd.DataFrame({ "Feature": feature_names, "Importance": importances }).sort_values("Importance", ascending=False).head(top_n)

fig = px.bar(
        importance_df.sort_values("Importance"),
        x="Importance", y="Feature", orientation="h",
        title=f"Top {top_n} Features Contributing to Risk"
    )
    return fig
else:
    st.warning("Model does not support feature importance.")
    return None

--- App Layout ---

st.title("🇿🇦 South African Municipal Protest Risk Predictor") st.markdown("This tool assesses protest risk probability based on municipal characteristics.")

model = load_model()

if model is not None: # === Single Prediction Section === st.header("Single Municipality Prediction")

with st.form("single_prediction"):
    cols = st.columns(3)
    province = cols[0].selectbox("Province", PROVINCES)
    district = cols[1].text_input("District Municipality")
    local_muni = cols[2].text_input("Local Municipality")

    st.subheader("Population Demographics")
    demo_cols = st.columns(5)
    total_population = demo_cols[0].number_input("Total Population", min_value=0, value=100000)
    black = demo_cols[1].number_input("Black African", min_value=0, value=80000)
    coloured = demo_cols[2].number_input("Coloured", min_value=0, value=10000)
    indian = demo_cols[3].number_input("Indian/Asian", min_value=0, value=5000)
    white = demo_cols[4].number_input("White", min_value=0, value=5000)

    st.subheader("Living Conditions")
    hs_cols = st.columns([2,1,1,1,1])
    informal = hs_cols[0].number_input("Informal Dwellings", min_value=0, value=5000)
    piped_water = hs_cols[1].number_input("Piped Water Access", min_value=0, value=70000)
    no_water = hs_cols[2].number_input("No Water Access", min_value=0, value=10000)
    pit_toilet = hs_cols[3].number_input("Pit Toilets", min_value=0, value=20000)
    bucket_toilet = hs_cols[4].number_input("Bucket Toilets", min_value=0, value=1000)

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    input_data = {
        'Province name': province,
        'District municipality name': district,
        'District/Local municipality name': local_muni,
        'Local municipality code': 0,
        'ID': 0,
        'Total Population': total_population,
        'Black African': black,
        'Coloured': coloured,
        'Indian/Asian': indian,
        'White': white,
        'Informal Dwelling': informal,
        'Piped (tap) water on community stand': piped_water,
        'No access to piped (tap) water': no_water,
        'Pit toilet': pit_toilet,
        'Bucket toilet': bucket_toilet,
    }
    input_df = pd.DataFrame([input_data])
    input_df = validate_inputs(input_df)

    risk_prob = model.predict_proba(input_df)[0][1] * 100
    st.success(f"Predicted Protest Risk: {risk_prob:.1f}%")

    col1, col2 = st.columns(2)
    col1.metric("Risk Level", "High" if risk_prob > 70 else "Medium" if risk_prob > 40 else "Low")
    col2.progress(int(risk_prob))

    fig = get_top_features(model, input_df.columns)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

# === Batch Prediction Section ===
st.header("Batch Prediction via CSV")

sample_template = pd.DataFrame(columns=[
    'Province name', 'District municipality name', 'District/Local municipality name',
    'Total Population', 'Black African', 'Coloured', 'Indian/Asian', 'White',
    'Informal Dwelling', 'Piped (tap) water on community stand',
    'No access to piped (tap) water', 'Pit toilet', 'Bucket toilet'
])

st.download_button(
    "Download CSV Template",
    sample_template.to_csv(index=False),
    "template.csv", "text/csv"
)

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df = validate_inputs(df)

        missing_cols = [col for col in sample_template.columns if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            predictions = model.predict_proba(df[sample_template.columns])[:, 1] * 100
            df['Protest Risk (%)'] = predictions.round(1)
            st.success(f"Processed {len(df)} records")

            st.subheader("Highest Risk Municipalities")
            st.dataframe(df.sort_values('Protest Risk (%)', ascending=False).head(5))

            csv = df.to_csv(index=False)
            st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

else: st.error("Model could not be loaded. Please check the file path and try again.")

Footer

st.markdown("---") st.caption("Protest Risk Prediction Model

