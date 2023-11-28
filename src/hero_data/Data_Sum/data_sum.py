""" This python file is for data summary in Hero analysis"""
import pandas as pd


class Reader:
    def __init__(self, path):
        self.path = path

    def combiner(self):
        # Read CSV files
        df1 = pd.read_csv(self.path[0])
        df2 = pd.read_csv(self.path[1])
        df1["Comic"] = "DC"
        df2["Comic"] = "MARVEL"

        # Assign the new column names to the DataFrame
        df2.columns = df1.columns

        # Concatenate DataFrames
        combined_df = pd.concat([df1, df2], ignore_index=True)
        return combined_df

    def reader(self):
        df = pd.read_csv(self.path, encoding="latin-1")
        return df
