""" This python file is for EDA for each column in Hero analysis"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class EDA_per_column:
    def __init__(self, df):
        """
        Initialize EDA_per_column class with a DataFrame.

        Args:
        - df (DataFrame): Input DataFrame for analysis.
        """
        self.df = df

    def cat_percentage(self, column):
        """
        Calculate the percentage distribution of categorical values in a column.

        Args:
        - column (str): Column name in the DataFrame.

        Returns:
        - Percentage distribution of categorical values in the column.
        """
        obj = round(
            self.df[column].value_counts() / self.df[column].value_counts().sum() * 100,
            2,
        )
        return obj

    def bar_chart(self, column):
        """
        Create a bar chart showing the frequency distribution of a column using Matplotlib and Seaborn.

        Args:
        - column (str): Column name in the DataFrame.
        """

        # Create the value counts
        value_counts = self.df[column].value_counts()

        if value_counts.count() > 6:
            rot_num = 60
        else:
            rot_num = 25

        # Create a figure with two subplots
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Original Matplotlib plot
        value_counts.plot(
            kind="bar",
            ax=ax1,
            rot=rot_num,
            title=f"Freq Distribution of {column} of heroes (Matplotlib)",
            xlabel="",
            edgecolor="black",
        )

        # Adding labels to the bars
        for i, v in enumerate(value_counts):
            ax1.text(i, v, str(v), ha="center", va="bottom", fontsize=8)

        # Seaborn plot
        sns.barplot(
            x=value_counts.index, y=value_counts.values, ax=ax2, edgecolor="black"
        )
        ax2.set_title(f"Freq Distribution of {column} of heroes (Seaborn)")

        # Adding labels to the bars
        for i, v in enumerate(value_counts):
            ax2.text(i, v, str(v), ha="center", va="bottom", fontsize=8)

        # Rotate x-axis labels in Seaborn
        ax2.set_xticklabels(value_counts.index, rotation=rot_num)

        plt.xlabel("")
        # Display the plot
        plt.tight_layout()
        plt.show()

    def barh_chart(self, column):
        """
        Create a horizontal bar chart showing the frequency distribution of a column using Matplotlib and Seaborn.

        Args:
        - column (str): Column name in the DataFrame.
        """
        # Create the value counts
        value_counts = self.df[column].value_counts()

        # Create a figure with two subplots
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Original Matplotlib horizontal bar plot
        value_counts.plot(
            kind="barh",
            ax=ax1,
            title=f"Freq Distribution of {column} of heroes (Matplotlib)",
            xlabel="",
        )

        # Adding labels to the bars
        for i, v in enumerate(value_counts):
            ax1.text(v, i, str(v), ha="left", va="center", fontsize=8)
        ax1.invert_yaxis()
        # Seaborn horizontal bar plot
        sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax2)
        ax2.set_title(f"Freq Distribution of {column} of heroes (Seaborn)")

        # Adding labels to the bars
        for i, v in enumerate(value_counts):
            ax2.text(v, i, str(v), ha="left", va="center", fontsize=8)

        # Rotate y-axis labels in Seaborn
        ax2.set_yticklabels(value_counts.index)

        plt.xlabel("")
        # Display the plot
        plt.tight_layout()
        plt.show()

    def hist_chart(self, column):
        """
        Create a histogram displaying the distribution of a numerical column using Matplotlib and Seaborn.

        Args:
        - column (str): Column name in the DataFrame.
        """
        # Create a figure with two subplots
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Original Matplotlib histogram plot
        hist, bins, _ = ax1.hist(self.df[column], bins=10, edgecolor="black")
        ax1.set_title(f"Histogram of {column} (Matplotlib)")

        # Adding labels to the bars for Matplotlib
        for i in range(len(hist)):
            ax1.text(bins[i] + 0.5, hist[i], str(int(hist[i])))

        # Seaborn histogram plot
        sns.histplot(data=self.df, x=column, bins=10, kde=False, ax=ax2)
        ax2.set_title(f"Histogram of {column} (Seaborn)")
        for i in range(len(hist)):
            ax2.text(
                bins[i] + 0.5,
                hist[i],
                str(int(hist[i])),
                ha="left",
                va="center",
                fontsize=8,
            )

        plt.tight_layout()
        plt.show()

    def line_chart(self, column):
        """
        Create a line plot showing the frequency distribution of a column using Matplotlib and Seaborn.

        Args:
        - column (str): Column name in the DataFrame.
        """
        # Create the value counts
        value_counts = self.df[column].value_counts().sort_index()

        # Create a figure with two subplots
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Original Matplotlib line plot
        ax1.plot(value_counts.index, value_counts.values, marker="o")
        ax1.set_title(f"Line Plot of {column} (Matplotlib)")
        ax1.set_xlabel(column)
        ax1.set_ylabel("Frequency")

        # Seaborn line plot
        sns.lineplot(x=value_counts.index, y=value_counts.values, ax=ax2)
        ax2.set_title(f"Line Plot of {column} (Seaborn)")
        ax2.set_xlabel(column)
        ax2.set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    def boxplot_chart(self, column):
        """
        Create a box plot to visualize the distribution of a column using Matplotlib and Seaborn.

        Args:
        - column (str): Column name in the DataFrame.
        """
        # Create a figure with two subplots
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Original Matplotlib box plot
        ax1.boxplot(self.df[column])
        ax1.set_title(f"Box Plot of {column} (Matplotlib)")
        ax1.set_xlabel(column)
        ax1.set_ylabel("Frequency")

        # Seaborn box plot
        sns.boxplot(data=self.df, x=column, ax=ax2)
        ax2.set_title(f"Box Plot of {column} (Seaborn)")
        ax2.set_xlabel(column)
        ax2.set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()
