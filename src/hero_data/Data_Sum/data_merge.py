import pandas as pd


class Merger:
    """
    merge datasets
    """

    def __init__(self, df1, df2):
        self.df1 = df1
        self.df2 = df2

    def join(self):
        """
        join datasets
        """
        match = pd.read_csv(
            "https://raw.githubusercontent.com/jc000222/Hero_Analysis/main/match.csv"
        )
        merged_2_3 = pd.merge(match, self.df2, on="title", how="left")
        final_result = pd.merge(
            self.df1, merged_2_3, left_on="name", right_on="name", how="inner"
        )
        return final_result
