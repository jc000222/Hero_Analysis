"""
Module Name: data_sum
This module is for reading data for data summary
Author: Ruoyu Chen
"""
import pandas as pd


class Reader:
    def __init__(self, path):
        """
        Initialize the Reader class with a file path.

        Args:
        - path (str or list): File path(s) for reading CSV file(s).
        """
        self.path = path

    def combiner(self):
        """
        Combine two CSV files from different publishers into a single DataFrame.

        Returns:
        - combined_df (DataFrame): Merged DataFrame with an additional 'Comic' column.
        """
        df1 = pd.read_csv(self.path[0])
        df2 = pd.read_csv(self.path[1])
        df1["Comic"] = "DC"
        df2["Comic"] = "MARVEL"
        df2.columns = df1.columns
        combined_df = pd.concat([df1, df2], ignore_index=True)
        return combined_df

    def reader(self):
        """
        Read a CSV file.

        Returns:
        - df (DataFrame): Read DataFrame from the specified path.
        """
        df = pd.read_csv(self.path, encoding="latin-1")
        return df
