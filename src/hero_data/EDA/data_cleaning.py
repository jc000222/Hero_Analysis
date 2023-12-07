"""
Module Name: data_cleaning
This python file is for cleaning data in Hero analysis
Author: Ruoyu Chen
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


class Cleaner:
    """
    This class is for cleaning data in in the EDA
    """

    def __init__(self, df):
        """
        Initialize the Clean class with a DataFrame.

        Args:
        - df (DataFrame): Input DataFrame to be cleaned.
        """
        self.df = df
        pd.options.mode.chained_assignment = None
        warnings.filterwarnings("ignore", category=UserWarning)

    def null_check(self):
        """
        Perform null check and display missing values heatmap.

        Returns:
        - Null counts for each column in the DataFrame.
        """
        plt.rcParams["figure.figsize"] = (15, 6)
        sns.heatmap(self.df.isnull(), yticklabels=False, cbar=False)
        plt.title("Missing null values")
        plt.xticks(rotation=30)
        return self.df.isnull().sum().sort_values(ascending=False)

    def delete_columns(self, columns_to_exclude):
        """
        Delete specified columns from the DataFrame.

        Args:
        - columns_to_exclude (list): List of columns to exclude from the DataFrame.

        Returns:
        - Cleaned DataFrame after removing specified columns.
        """
        self.df = self.df[
            [col for col in self.df.columns if col not in columns_to_exclude]
        ]
        return self.df

    def replace_null(self):
        """
        Replace null values in specific columns with predefined values.

        Returns:
        - Cleaned DataFrame after handling null values.
        """
        self.df["ID"].fillna("Unknown", inplace=True)
        self.df["ID"] = self.df["ID"].replace("Identity Unknown", "Unknown")
        self.df["ALIGN"].fillna("Unknown", inplace=True)
        self.df["SEX"].fillna("Unknown", inplace=True)
        self.df["GSM"].fillna("Not GSM", inplace=True)
        self.df = self.df.dropna(subset=["ALIVE"])
        # drop the hero that has no first appearance
        self.df = self.df.dropna(subset=["YEAR"])
        # fill the appearance of the heroes that has a first appearance year but has no APPEARANCES times with 1 time.
        self.df["APPEARANCES"].fillna(1, inplace=True)
        self.df["APPEARANCES"] = self.df["APPEARANCES"].astype(int)
        self.df["YEAR"] = self.df["YEAR"].astype(int)
        return self.df

    def rename(self):
        """
        Rename specific columns with other values.

        Returns:
        - DataFrame after renamed columns.
        """

        # Changing column names
        self.df.rename(columns={"old_column_name": "new_column_name"}, inplace=True)
        return self.df
