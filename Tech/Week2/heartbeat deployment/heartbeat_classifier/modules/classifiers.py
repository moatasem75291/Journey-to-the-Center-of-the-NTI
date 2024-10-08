import numpy as np
from xgboost import XGBClassifier


class Classifier:
    def __init__(self, binary_model_path: str, multi_model_path: str):
        """
        Initialize the classifier with the model paths.

        Parameters:
        - binary_model_path: str
        - multi_model_path: str
        """
        self.binary_model = XGBClassifier()
        self.binary_model.load_model(binary_model_path)

        self.multi_model = XGBClassifier()
        self.multi_model.load_model(multi_model_path)

        self.class_names = {
            0: "Normal Beats",
            1: "Supraventricular Ectopy Beats",
            2: "Ventricular Ectopy Beats",
            3: "Fusion Beats",
            4: "Unclassifiable Beats",
        }

    def classify_binary(self, data: np.array) -> int:
        """
        Perform binary classification.

        Parameters:
        - data: np.array

        Returns:
        - Prediction: int (0 for normal, 1 for abnormal)
        """
        prediction = self.binary_model.predict(data)[0]
        return prediction

    def classify_multi(self, data: np.array) -> str:
        """
        Perform multi-class classification.

        Parameters:
        - data: np.array

        Returns:
        - Class label: str
        """
        multi_pred = self.multi_model.predict(data)[0] + 1
        return self.class_names[multi_pred]
