"""This python file do the EDA EDA_multi_column work for hero analysis"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class EDA_multi_column:
    def __init__(self, df):
        """
        Initialize the EDA_multi_column class with a DataFrame.

        Args:
        - df (DataFrame): Input DataFrame for analysis.
        """
        self.df = df

    def bar_chart_m(self, columns):
        """
        Generate a bar chart based on two columns.

        Args:
        - columns (list): List of two columns to be plotted.
        """
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        multi_column = pd.crosstab(
            index=self.df[columns[0]], columns=self.df[columns[1]]
        )
        multi_column.plot.bar(
            ax=ax1,
            rot=0,
            title=f"Freq Distribution of {columns[1]} on {columns[1]} of heroes (Matplotlib)",
            xlabel="",
            edgecolor="black",
        )

        sns.countplot(
            x=self.df[columns[0]],
            hue=self.df[columns[1]],
            data=self.df,
            ax=ax2,
            edgecolor="black",
            order=self.df[columns[0]].value_counts().index,
        )
        ax2.set_title(
            f"Freq Distribution of {columns[1]} on {columns[1]} of heroes (Seaborn)"
        )

        plt.xlabel("")
        plt.tight_layout()
        plt.show()

    def scatter_plot(self, columns):
        """
        Generate a scatter plot based on two columns.

        Args:
        - columns (list): List of two columns for x and y axes.
        """
        plt.figure(figsize=(10, 6))

        sns.scatterplot(data=self.df, x=columns[0], y=columns[1])

        plt.title(f"Scatter Plot of {columns[0]} vs {columns[1]}")
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])

        plt.tight_layout()
        plt.show()
