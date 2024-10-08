import pandas as pd


class DataPreprocessor:
    def preprocess(self, data: pd.DataFrame, num_samples: int) -> pd.DataFrame:
        """
        Preprocess the input data by selecting 5 rows from each category (based on the last column)
        and selecting the first 100 features (columns) used for training.

        Parameters:
        - data: pd.DataFrame

        Returns:
        - Preprocessed data with first 100 features and 5 samples per category.
        """
        category_column = data.columns[-1]

        # shuffle data before selection
        data = data.sample(frac=1).reset_index(drop=True)
        sampled_data = (
            data.groupby(category_column)
            .apply(lambda x: x.sample(min(num_samples, len(x))))
            .reset_index(drop=True)
        )
        processed_data = sampled_data.iloc[:, :100]

        return processed_data
