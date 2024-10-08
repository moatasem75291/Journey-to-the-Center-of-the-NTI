import streamlit as st
import pandas as pd
from modules.preprocessing import DataPreprocessor
from modules.classifiers import Classifier
from modules.ui import DashboardUI, HelpUI

binary_model_path = "models/binary_classifier_model.json"
multi_model_path = "models/multiclass_classifier_model.json"

classifier = Classifier(binary_model_path, multi_model_path)
preprocessor = DataPreprocessor()

menu = ["Help", "Dashboard"]
choice = st.sidebar.selectbox("Select a Page", menu)

if choice == "Dashboard":
    st.title("Patient Classification Dashboard")

    uploaded_file = DashboardUI.upload_file()

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, header=None)
        num_samples = st.number_input(
            "Enter the number of samples from each Class to classify:",
            min_value=1,
            max_value=len(data),
            # value=5,
        )
        if num_samples is not None:
            preprocessed_data = preprocessor.preprocess(data, num_samples=num_samples)

            DashboardUI.display_dashboard(preprocessed_data, classifier, data)

elif choice == "Help":
    HelpUI.display_help()
