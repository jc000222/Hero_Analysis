"""Run Hero_data CLI"""
from hero_data.Data_Sum.data_sum import Reader


def main():
    """
    Run Apex analysis as a script. Choose which part to run.
    """
    print(
        "Hero Analysis:-Project 3\nAuthor: Ruoyu Chen\nGithub Repository: https://github.com/jc000222/Hero_analysis"
    )

    while True:
        print("-----------------------------------")
        print(
            "(1)Part 1: Data summary\n(2)Part 2: Using an API\n(3)Part 3: Web Scraping\n(4)Part 4: Analyze dataset"
        )
        print("(5)Extra: Selenium\n(0)Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            Datasummary()
        elif choice == "0":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a valid option.")


def Datasummary():
    data_url = [
        "https://raw.githubusercontent.com/fivethirtyeight/data/master/comic-characters/dc-wikia-data.csv",
        "https://raw.githubusercontent.com/fivethirtyeight/data/master/comic-characters/marvel-wikia-data.csv",
    ]

    reader_comic = Reader(data_url)
    df_comic_raw = reader_comic.combiner()
    print(df_comic_raw.head())

    data = "hero_data/data/marvelvsdc.csv"

    reader_movie = Reader(data)
    df_movie_raw = reader_movie.reader()
    print(df_movie_raw.head())
