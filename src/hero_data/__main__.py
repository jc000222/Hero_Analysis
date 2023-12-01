"""Enable running `python -m chip_analysis_package`."""

from hero_data.Data_Sum.data_sum import Reader
from hero_data.EDA.per_colum import EDA_Per_Column
from hero_data.EDA.data_cleaning import Cleaner
from hero_data.Infer import Analyzer
from src.hero_data.EDA.multi_coumun import EDA_Multi_Column


def main():
    """
    Run Apex analysis as a script. Choose which part to run.
    """
    print(
        "Hero Analysis:-Project 3\nAuthor: Ruoyu Chen\nGithub Repository: https://github.com/jc000222/Hero_analysis"
    )

    print("data summary________________________________________________________")

    data_url = [
        "https://raw.githubusercontent.com/fivethirtyeight/data/master/comic-characters/dc-wikia-data.csv",
        "https://raw.githubusercontent.com/fivethirtyeight/data/master/comic-characters/marvel-wikia-data.csv",
    ]

    reader_comic = Reader(data_url)
    df_comic_raw = reader_comic.combiner()
    print("this is raw data1--------------")
    print(df_comic_raw.head())

    data = (
        "https://raw.githubusercontent.com/jc000222/Hero_Analysis/main/marvelvsdc.csv"
    )

    reader_movie = Reader(data)
    df_movie_raw = reader_movie.reader()
    df_movie_raw.head()
    print("this is raw data2--------------")
    print(df_movie_raw.head())

    print("EDA________________________________________________________")
    eda_comics = EDA_Per_Column(df_comic_raw)
    eda_comics.bar_chart("ALIGN")
    eda_comics.barh_chart("GSM")
    eda_comics_m = EDA_Multi_Column(df_comic_raw)
    eda_comics_m.bar_chart_m(["ALIVE", "ALIGN"])
    eda_movies = EDA_Per_Column(df_movie_raw)
    eda_movies.boxplot_chart("runtime")

    print("EDA clean________________________________________________________")
    cleaner_comic = Cleaner(df_comic_raw)

    columns_to_exclude = ["page_id", "urlslug", "EYE", "HAIR", "FIRST APPEARANCE"]
    cleaner_comic.delete_columns(columns_to_exclude)

    cleaner_movie = Cleaner(df_movie_raw)

    columns_to_exclude = [
        "Unnamed: 0",
        "mpa_rating",
        "imdb_votes",
        "director",
        "stars",
        "description",
        "crit_consensus",
        "tomato_review",
        "tom_ratings",
    ]
    df_movie_c = cleaner_movie.delete_columns(columns_to_exclude)
    df_comic = cleaner_comic.replace_null()
    df_movie = df_movie_c
    print(df_comic.head())
    print(df_movie.head())
    print("inference________________________________________________________")
    analyzer_comic = Analyzer(df_comic)
    analyzer_comic.Align_Sex()
    analyzer_comic.GSM_Comic()
    analyzer_movie = Analyzer(df_movie)
    analyzer_movie.genre_analysis()


main()
