import streamlit as st
import pandas as pd


class DashboardUI:
    @staticmethod
    def upload_file():
        """
        Handle file uploading and return the uploaded file.

        Returns:
        - uploaded_file: File
        """
        return st.file_uploader("Upload your input CSV file", type=["csv"])

    @staticmethod
    def display_dashboard(preprocessed_data, classifier, data):
        """
        Display the dashboard, show classification results in a table.

        Parameters:
        - preprocessed_data: pd.DataFrame
        - classifier: Classifier object
        - data: pd.DataFrame (original input data)
        """
        results = []
        for idx, _ in preprocessed_data.iterrows():
            row_preprocessed = preprocessed_data.iloc[idx, :].values.reshape(1, -1)
            binary_prediction = classifier.classify_binary(row_preprocessed)
            if binary_prediction == 0:
                prediction = "Normal"
            else:
                multi_prediction = classifier.classify_multi(row_preprocessed)
                prediction = multi_prediction

            results.append((idx, prediction))

        results_df = pd.DataFrame(results, columns=["Index", "Prediction"])

        DashboardUI.display_colored_table(results_df)

    @staticmethod
    def display_colored_table(results_df):
        """
        Display a color-coded table based on the prediction value.

        Parameters:
        - results_df: pd.DataFrame containing index and prediction values
        """

        def color_code(row):
            """
            Define the color for each class.
            """
            if row["Prediction"] == "Normal":
                return ["background-color: #4a8735"] * 2
            elif row["Prediction"] == "Supraventricular Ectopy Beats":
                return ["background-color: #ed312d"] * 2
            elif row["Prediction"] == "Ventricular Ectopy Beats":
                return ["background-color: #e3c340"] * 2
            elif row["Prediction"] == "Fusion Beats":
                return ["background-color: #56b1f0"] * 2
            elif row["Prediction"] == "Unclassifiable Beats":
                return ["background-color: #b379ed"] * 2
            else:
                return [""] * 2

        styled_df = results_df.style.apply(color_code, axis=1)

        st.dataframe(styled_df, hide_index=True, use_container_width=True)


import streamlit as st


class HelpUI:
    @staticmethod
    def display_help():
        """
        Display the help and instructions page.
        """
        st.title("📋 Help and Information")

        st.header("🌟 Project Overview")
        st.write(
            """
            Welcome to the **Patient Classification System**! This tool is designed to help healthcare professionals classify cardiac data in two stages:

            1️⃣ **Binary Classification**: The model first determines if a patient is **Normal (0)** or **Abnormal (1)**.
            
            2️⃣ **Multi-Class Classification**: If abnormal, the model further classifies into one of the following categories:
            - 🟥 **Supraventricular Ectopy Beats (S)**
            - 🟨 **Ventricular Ectopy Beats (V)**
            - 🟦 **Fusion Beats (F)**
            - 🟪 **Unclassifiable Beats (Q)**
            
            This system aims to assist with early detection of irregular cardiac patterns to improve patient care. 
            """
        )

        # Adding an image for better visualization
        st.image(
            "https://ars.els-cdn.com/content/image/1-s2.0-S0010482520302237-fx1.jpg",
            caption="Patient Classification Process",
            use_column_width=True,
        )

        st.header("🛠️ How to Use the Dashboard")
        st.write(
            """
            1️⃣ **Upload a CSV File** 📁:
               - The CSV file should contain patient data with the necessary features. Make sure the format matches what was used in training.

            2️⃣ **Classification Results** 🩺:
               - After uploading, the system will first determine if the patient is **Normal** or **Abnormal**.
               - If **Abnormal**, the system will provide a detailed classification of the abnormal cardiac beats.

            3️⃣ **Interpretation** 🔍:
               - For **Normal** patients, you’ll see a green success message.
               - For **Abnormal** patients, you’ll see the specific abnormal beat type with color-coded results.

            4️⃣ **Repeat for New Data** 🔄:
               - You can upload new CSV files anytime to analyze other patient data.
            """
        )

        st.header("ℹ️ Additional Information")
        st.write(
            """
            - Ensure the CSV file is correctly formatted and contains all necessary data.
            - The classification models are pre-trained, so the features in the CSV file should align with the training set.
            - 🛡️ **Data Privacy**: All patient data is handled securely within the app, and nothing is stored outside the system.
            """
        )

        st.sidebar.info(
            "Need help or have questions? 📧 Contact us at moatasem.mohammed226@gmail.com"
        )
