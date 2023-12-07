"""This module is for the model building part for hero analysis"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd


class ModelBuilder:
    """
    This class is to build, train, and visualize a logistic regression model for predicting 'ALIGN'.

    This class allows filtering a DataFrame based on specific conditions,
    preprocessing the data, building a logistic regression model, and visualizing
    the relationship between features and predicted probabilities.

    Attributes:
    - df (DataFrame): Input DataFrame for modeling.
    - filtered_df (DataFrame): Filtered DataFrame based on specific conditions.
    - preprocessor (ColumnTransformer): Preprocessor for transforming categorical columns.
    - pipeline (Pipeline): Trained pipeline containing the logistic regression model.
    - X (DataFrame): Predictor variables.
    - y (array): Target variable.
    - X_test (DataFrame): Test set for evaluation.
    """

    def __init__(self, df):
        """
        Initialize the ModelBuilder class.

        Args:
        - df (DataFrame): Input DataFrame for modeling.
        """
        self.df = df
        self.filtered_df = self.data_select()

    def data_select(self):
        """
        Filter the DataFrame based on specific conditions.

        Returns:
        - filtered_df (DataFrame): Filtered DataFrame.
        """
        filtered_df = self.df[
            self.df["SEX"].isin(["Female Characters", "Male Characters"])
            & self.df["ID"].isin(["Secret Identity", "Public Identity"])
            & self.df["ALIGN"].isin(["Bad Characters", "Good Characters"])
        ]
        return filtered_df

    def model_ALIGN(self):
        """
        Build and train a model to predict 'ALIGN'.

        Returns:
        - pipeline (Pipeline): Trained pipeline containing the model.
        """
        categorical_cols = ["ID", "ALIVE", "SEX"]
        numerical_cols = ["APPEARANCES"]
        predictors = ["ID", "ALIVE", "SEX", "APPEARANCES"]
        target = ["ALIGN"]
        preprocessor = ColumnTransformer(
            transformers=[
                ("ordinal", OrdinalEncoder(), categorical_cols),
                ("numeric", "passthrough", numerical_cols),
            ]
        )
        self.preprocessor = preprocessor
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())]
        )

        self.X = self.filtered_df[predictors]
        y = self.filtered_df[target]
        self.y = np.ravel(y.values)

        cv_scores = cross_val_score(pipeline, self.X, self.y, cv=5)

        print("Cross-validation scores:", cv_scores)
        print("Mean accuracy:", cv_scores.mean())
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )
        self.y_test = y_test
        self.X_test = X_test
        pipeline.fit(X_train, y_train)
        self.X_test = X_test
        self.pipeline = pipeline
        return pipeline

    def view_data(self):
        """
        View the transformed data.

        Returns:
        - transformed_df (DataFrame): Transformed DataFrame.
        """
        predictors = ["ID", "ALIVE", "SEX", "APPEARANCES"]
        transformed_X = self.preprocessor.transform(self.X)
        transformed_df = pd.DataFrame(transformed_X, columns=predictors)
        return transformed_df.head(10)

    def coefficient(self):
        """
        Display logistic regression coefficients.

        Returns:
        - coef (array): Coefficients of the features.
        """
        predictors = ["ID", "ALIVE", "SEX", "APPEARANCES"]
        coef = self.pipeline.named_steps["classifier"].coef_
        print("Logistic Regression Coefficients:")
        for predictor, coef in zip(predictors, coef[0]):
            print(f"{predictor}: {coef}")
        return coef

    def chart(self):
        """
        Visualize the relationship between features and predicted probabilities.
        """
        probabilities = self.pipeline.predict_proba(self.X_test)
        positive_class_probabilities = probabilities[:, 1]
        features = ["ID", "ALIVE", "SEX", "APPEARANCES"]

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs = axs.flatten()

        for i, feature in enumerate(features):
            axs[i].scatter(
                self.X_test[feature], positive_class_probabilities, alpha=0.08
            )
            axs[i].set_xlabel(feature)
            axs[i].set_ylabel("Predicted probabilities")
            axs[i].set_title(
                f"Logistic Regression: {feature} vs Predicted probabilities"
            )
            axs[i].set_ylim(0, 1)  # Set y-axis range from 0 to 1

        plt.tight_layout()
        plt.show()
