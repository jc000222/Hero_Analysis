"""This module is for merging data for data summary"""
import pandas as pd


class Merger:
    """
    A class to merge datasets based on specified conditions.

    Attributes:
        df1 (DataFrame): First DataFrame for merging.
        df2 (DataFrame): Second DataFrame for merging.

    Methods:
        join(): Merges the specified DataFrames based on the defined conditions.
    """

    def __init__(self, df1, df2):
        """
        Initialize Merger with provided DataFrames for merging.

        Args:
        df1 (DataFrame): First DataFrame for merging.
        df2 (DataFrame): Second DataFrame for merging.
        """
        self.df1 = df1
        self.df2 = df2

    def join(self):
        """
        Performs a join operation on the specified DataFrames.

        Returns:
        DataFrame: The resulting merged DataFrame based on the defined conditions.
        """
        match = pd.read_csv(
            "https://raw.githubusercontent.com/jc000222/Hero_Analysis/main/match.csv"
        )
        merged_2_3 = pd.merge(match, self.df2, on="title", how="left")
        final_result = pd.merge(
            self.df1, merged_2_3, left_on="name", right_on="name", how="inner"
        )
        return final_result
