"""Run Hero_data CLI"""
from hero_data.Data_Sum.data_sum import Reader
from hero_data.EDA.per_colum import EDA_per_column
from hero_data.EDA.data_cleaning import Clean
from hero_data.Infer import Analysis


def main():
    """
    Run Apex analysis as a script. Choose which part to run.
    """
    print(
        "Hero Analysis:-Project 3\nAuthor: Ruoyu Chen\nGithub Repository: https://github.com/jc000222/Hero_analysis"
    )

    while True:
        print("-----------------------------------")
        print("(1)Part 1: Data summary\n(2)Part 2: EDA\n(3)clean\n(4)Inference")
        print("(0)Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            Datasummary()
        elif choice == "2":
            EDA()
        elif choice == "3":
            clean()
        elif choice == "4":
            Inf()
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
    while 1:
        data = input("Enter the csv: (input 0 to skip)")
        if data == "0":
            break
    reader_movie = Reader(data)
    df_movie_raw = reader_movie.reader()
    print(df_movie_raw.head())


def EDA():
    data_url = [
        "https://raw.githubusercontent.com/fivethirtyeight/data/master/comic-characters/dc-wikia-data.csv",
        "https://raw.githubusercontent.com/fivethirtyeight/data/master/comic-characters/marvel-wikia-data.csv",
    ]

    reader_comic = Reader(data_url)
    df_comic_raw = reader_comic.combiner()
    eda_comics = EDA_per_column(df_comic_raw)
    while True:
        print(df_comic_raw.columns)
        column = input("enter coloumn name")
        print(df_comic_raw[column].describe())
        choice = input("enter chart type:1bar, 2bar horizon, 3hist, 4line")
        if choice == "1":
            eda_comics.bar_chart(column)
            break
        elif choice == "2":
            eda_comics.barh_chart(column)
            break
        elif choice == "3":
            eda_comics.hist_chart(column)
            break
        elif choice == "4":
            eda_comics.line_chart(column)
            break
        elif choice == "0":
            print("Exiting...")
            break


def clean():
    data_url = [
        "https://raw.githubusercontent.com/fivethirtyeight/data/master/comic-characters/dc-wikia-data.csv",
        "https://raw.githubusercontent.com/fivethirtyeight/data/master/comic-characters/marvel-wikia-data.csv",
    ]

    reader_comic = Reader(data_url)
    df_comic_raw = reader_comic.combiner()
    cleaner_comic = Clean(df_comic_raw)

    columns_to_exclude = ["page_id", "urlslug", "EYE", "HAIR", "FIRST APPEARANCE"]
    df_comic_c = cleaner_comic.delete_columns(columns_to_exclude)
    print(df_comic_c.columns)
    cleaner_comic.null_check()
    df_comic = cleaner_comic.replace_null()
    cleaner_comic.null_check()
    return df_comic


def Inf():
    df_comic = clean()
    analyzer_comic = Analysis(df_comic)
    analyzer_comic.Align_Sex()
    analyzer_comic.GSM_Comic()
