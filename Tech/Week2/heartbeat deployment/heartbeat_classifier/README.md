# Heartbeat Classification Dashboard

Welcome to the **Heartbeat Classification Dashboard**! This project is designed to help healthcare professionals classify patient cardiac data into normal or abnormal classes, and further classify abnormal cases into specific beat types.

## 🚀 Overview

The application uses machine learning models in two stages:

1. **Binary Classification**: First, it determines if a patient is either **Normal (0)** or **Abnormal (1)**.
2. **Multi-Class Classification**: If the patient is classified as abnormal, the system further identifies the type of abnormal beat:
   - 🟥 **Supraventricular Ectopy Beats (S)**
   - 🟨 **Ventricular Ectopy Beats (V)**
   - 🟦 **Fusion Beats (F)**
   - 🟪 **Unclassifiable Beats (Q)**

The project is built with **Streamlit** for the web-based interface and **XGBoost** models for classification tasks.

---

## 🌟 Features

- **File Upload**: Users can upload a CSV file containing patient data.
- **Classification**: The app classifies whether the patient is normal or abnormal based on their data.
- **Detailed Results**: For abnormal cases, the system will provide a further classification into specific cardiac conditions.
- **Color-coded Table**: Results are displayed in a color-coded table for better interpretation.
- **Help Section**: Includes a detailed help page to guide users on how to use the system.

---

## 📊 Classification Models

### Binary Classifier:

- This model classifies the input data into two categories:
  - **Normal (0)**: Green background.
  - **Abnormal (1)**: Proceeds to the multi-class classification.

### Multi-Class Classifier:

- For patients classified as abnormal, it further categorizes them into one of four classes:
  - 🟥 **Supraventricular Ectopy Beats (S)**: Red background.
  - 🟨 **Ventricular Ectopy Beats (V)**: Yellow background.
  - 🟦 **Fusion Beats (F)**: Blue background.
  - 🟪 **Unclassifiable Beats (Q)**: Purple background.

---

## 🛠️ How to Use the App

1. **Upload CSV File**: Click the "Upload your input CSV file" button and select a CSV file from your computer. Ensure that the file contains the appropriate patient data with the required features.
2. **Set the Sample Size**: Choose how many samples from the CSV file you'd like to classify.
3. **View Results**: The results will be displayed in a table format with color-coded predictions for each patient. The table will contain two columns:
   - **Index**: The row number from the CSV file.
   - **Prediction**: The predicted class (normal or one of the abnormal beat types).

---

## ⚙️ Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/moatasem75291/heartbeat_classifier.git
   cd heartbeat_classifier
   ```
2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

## 🖥️ Deployed Application

You can access the live application here: [Deployed Link](https://heartbeat-moatasem22.streamlit.app/)

## 📁 Project Structure

```bash
📂 heartbeat_classifier
│
├── 📁 models                     # Contains pre-trained model files
│   ├── binary_classifier_model.json
│   ├── multiclass_classifier_model.json
│
├── 📁 modules                    # Modules for preprocessing, classification, and UI
│   ├── classifiers.py
│   ├── preprocessing.py
│   ├── ui.py
│
├── app.py                        # Main Streamlit application file
├── requirements.txt              # List of required dependencies
└── README.md                     # This file
```

## 📑 Help Section

To learn more about how to use the app, click on the `Help` section in the sidebar once the app is running. This section provides a detailed overview of the project, step-by-step instructions for using the app, and contact information for support.

## 🤝 Contributing

Contributions are welcome! If you have any ideas or find bugs, feel free to open an issue or create a pull request.

## 📧 Contact

If you have any questions or issues, feel free to reach out to the project maintainer:

- Email: moatasem.mohammed226@gmail.com
