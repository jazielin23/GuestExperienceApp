import streamlit as st
import old_model_measurement, data_preparation,  ml_engine

# Green #0D3512
# Red #FF0000

# st.markdown("<h1 style='color: #0D3512; font-size: 40px; text-align:center; font-weight: normal;'> Customer Experience Data Analysis</h1>", unsafe_allow_html=True)
st.write("<h1 style='color: #0D3512; font-size: 40px; text-align:center; font-weight: normal;'> Customer Experience Data Analysis</h1>", unsafe_allow_html=True)
st.write("")
st.write("")

st.sidebar.write("<h1 style='color: #0D3512; font-size: 20px; text-align:center; font-weight: normal;'> Navigation Panel </h1>", unsafe_allow_html=True)

# page = st.sidebar.radio("Choose a task", [ "Data Preparation", "Old Model Measurement", "ML Engine"])


# if page == "Data Preparation":
#     data_preparation.run()
# elif page == "Old Model Measurement":
#     old_model_measurement.run()
# elif page == "ML Engine":
#     ml_engine.run()


# Initialize state
if "page" not in st.session_state:
    st.session_state.page = "Data Preparation"

# Sidebar navigation
page = st.sidebar.radio(
    "Choose a task", 
    ["Data Preparation", "Old Model Measurement", "ML Engine"], 
    index=["Data Preparation", "Old Model Measurement", "ML Engine"].index(st.session_state.page)
)
st.session_state.page = page

# Page routing
if page == "Data Preparation":
    data_preparation.run()
elif page == "Old Model Measurement":
    old_model_measurement.run()
elif page == "ML Engine":
    ml_engine.run()
