import pandas as pd


class Merger:
    def __init__(self, df1, df2, df3):
        self.df1 = df1
        self.df1 = df2
        self.df1 = df3

    def join(self):
        merged_2_3 = pd.merge(self.df2, self.df3, on="title", how="right")
        final_result = pd.merge(
            self.df1, merged_2_3, left_on="name", right_on="name", how="inner"
        )
        return final_result
