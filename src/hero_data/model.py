from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


class ModelBuilder:
    def __init__(self, df):
        self.df = df
        self.filtered_df = self.data_select()

    def data_select(self):
        filtered_df = self.df[
            self.df["SEX"].isin(["Female Characters", "Male Characters"])
            & self.df["ID"].isin(["Secret Identity", "Public Identity"])
            & self.df["ALIGN"].isin(["Bad Characters", "Good Characters"])
        ]
        return filtered_df

    def model_ALIGN(self):
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

        pipeline.fit(X_train, y_train)

        logreg_coef = pipeline.named_steps["classifier"].coef_
        print("Logistic Regression Coefficients:")
        for predictor, coef in zip(predictors, logreg_coef[0]):
            print(f"{predictor}: {coef}")

        probabilities = pipeline.predict_proba(X_test)
        positive_class_probabilities = probabilities[:, 1]

        print("Probabilities of the positive class:", positive_class_probabilities)

        features = ["ID", "ALIVE", "SEX", "APPEARANCES"]

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs = axs.flatten()

        for i, feature in enumerate(features):
            axs[i].scatter(X_test[feature], positive_class_probabilities, alpha=0.05)
            axs[i].set_xlabel(feature)
            axs[i].set_ylabel("Predicted probabilities")
            axs[i].set_title(
                f"Logistic Regression: {feature} vs Predicted probabilities"
            )
            axs[i].set_ylim(0, 1)  # Set y-axis range from 0 to 1

        plt.tight_layout()
        plt.show()
        return pipeline
