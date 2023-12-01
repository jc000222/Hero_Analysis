"""This python file do the inference work for hero analysis"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Analyzer:
    """
    This class is for analysis in the inference
    """

    def __init__(self, df):
        """
        Initialize the Clean class with a DataFrame.

        Args:
        - df (DataFrame): Input DataFrame to be cleaned.
        """
        self.df = df

    def genre_analysis(self):
        """
        Perform genre analysis on movie data.

        This method calculates average ratings across different genres and displays the count of movies in each genre.

        Displays a bar plot showing the average ratings and movie count per genre.
        """
        # Adjusting the IMDb ratings
        self.df["imdb_rating"] = self.df["imdb_rating"] * 10

        # Calculate the average rating considering IMDb, Tomato Meter, and Tomato Audience Score
        self.df["average_rating"] = self.df[
            ["imdb_rating", "tomato_meter", "tom_aud_score"]
        ].mean(axis=1)

        # Splitting genres into individual rows
        movies_melted = self.df.assign(genre=self.df["genre"].str.split(", ")).explode(
            "genre"
        )

        # Grouping by genre to get the average ratings and counts for each genre
        genre_avg = (
            movies_melted.groupby("genre")
            .agg(
                {
                    "imdb_rating": "mean",
                    "tomato_meter": "mean",
                    "tom_aud_score": "mean",
                    "average_rating": "mean",
                }
            )
            .reset_index()
        )

        # Count of movies in each genre
        genre_count = movies_melted["genre"].value_counts().reset_index()
        genre_count.columns = ["genre", "count"]

        # Merging average ratings and count values with genre counts
        genre_avg_count = pd.merge(genre_avg, genre_count, on="genre")

        # Melt the DataFrame to plot multiple values against the same x-axis
        genre_avg_count_melted = genre_avg_count.melt(
            id_vars="genre", var_name="Rating", value_name="Average Ratings"
        )

        # Plotting the average ratings and count values per genre
        ax = sns.barplot(
            data=genre_avg_count_melted,
            x="genre",
            y="Average Ratings",
            hue="Rating",
            palette="magma",
        )

        # Adding values on top of the average rating bars
        for p in ax.patches:
            if p.get_height() > 0:  # To prevent zero values being annotated
                ax.annotate(
                    f"{p.get_height():.0f}",
                    (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center",
                    va="center",
                    xytext=(0, 10),
                    textcoords="offset points",
                )

        # Adding the count information as a second y-axis
        ax2 = plt.twinx()
        sns.barplot(
            data=genre_avg_count,
            x="genre",
            y="count",
            ax=ax2,
            alpha=0.3,
            color="orange",
        )

        plt.title("Average Ratings and Movie Count per Genre")
        plt.xlabel("Genre")
        plt.ylabel("Movie Count")
        plt.tight_layout()
        plt.show()

    def Align_Sex(self):
        """
        Analyzes the alignment of characters based on their gender distribution using a horizontal stacked bar chart.

        Returns:
        Displays a horizontal stacked bar chart indicating the percentage distribution of gender across different character alignments.
        """
        # We create a cross table
        filtered_df = self.df[self.df["SEX"] != "Unknown"]

        AS = pd.crosstab(index=filtered_df["ALIGN"], columns=filtered_df["SEX"])

        # Adding a column with the total
        AS["Total"] = AS.sum(axis=1)

        # We normalize by obtaining the percentages in a way that each column (attribute) in the cross table adds up to 100%
        # Then multiply it's value by 100 and round it to two decimals, to clearly observe the percentages
        AS_normalized = round(AS.div(AS.sum(axis=0), axis=1) * 100, 2)

        # We set seaborn plotting aesthetics
        sns.set(style="white")

        # Define data and transpose the cross tab so that the application mode is in the X axis
        AS_normalized.T.plot(kind="barh", stacked=True)

        # Title and label
        plt.xlabel("Percentage of Alignment")
        plt.title("Percentage of SEX by Alignment")

    def GSM_Comic(self):
        """
        Analyzes the GSM of characters based on their gender distribution using a horizontal stacked bar chart.

        Returns:
        Displays a horizontal stacked bar chart indicating the percentage distribution of gender across different character alignments.
        """
        # Filter the DataFrame to exclude 'Not GSM' values
        filtered_df = self.df[self.df["GSM"] != "Not GSM"]

        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))

        # Create a count plot to display the distribution of GSM across comics (Marvel and DC)
        sns.countplot(data=filtered_df, x="GSM", hue="Comic")

        # Set title and labels
        plt.title("Number of Characters in Different GSM Categories - Marvel vs. DC")
        plt.xlabel("GSM")
        plt.ylabel("Count of Characters")

        # Show legend
        plt.legend(title="Comic")

        # Display the plot
        plt.xticks(rotation=25)  # Rotate x-axis labels for better readability
        plt.tight_layout()
        plt.show()
