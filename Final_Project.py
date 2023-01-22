from pyspark.sql.functions import *
from pyspark.sql.types import StringType, DateType,StructField, StructType, BooleanType
from pyspark.ml.feature import StopWordsRemover, RegexTokenizer
import pyspark.sql.functions as F
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import pandas as pd
import os
import sys
from datetime import datetime
from pathlib import Path


os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

if len(sys.argv) != 4:
  raise Exception("Exactly 3 arguments are required: <inputUri> <outputUri>")

inputUri1=sys.argv[1]
inputUri2=sys.argv[2]
outputUri=sys.argv[3]

# Create a Spark configuration and context
conf = SparkConf().setAppName("WorldCupTweets").setMaster('local[4]')
# conf = SparkConf().setAppName("WorldCupTweets").setMaster('local[4]')

sc = SparkContext(conf=conf)

# Create a SparkSession
spark = SparkSession.builder.appName("WorldCupTweets").getOrCreate()

# Read the CSV file as a DataFrame
df_init = pd.read_csv(inputUri1)

# remove null dates
df_init = df_init[~df_init['date'].isnull()]
df_schema = StructType([StructField("user_name", StringType(), True)\
                       ,StructField("user_location", StringType(), True)\
                       ,StructField("user_description", StringType(), True)\
                       ,StructField("date", StringType(), True)\
                       ,StructField("text", StringType(), True)\
                       ,StructField("hashtags", StringType(), True)\
                        ])


tweets_df = spark.createDataFrame(df_init,df_schema)

def df_preproc(df):

    # add new column named 'day' and include ONLY the date of the tweet eg. 2022/12/11 and make it DateType
    extract_date = udf(lambda x: x.split()[0], StringType())
    df = tweets_df.withColumn("day", extract_date(col("date")))

    # sort data by 'day'
    df = df.sort(col("day").desc())

    # process the 'text' column and filter out any symbols, stopwords and duplicate words
    # RegexTokenizer to tokenize text
    regex_tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
    df = regex_tokenizer.transform(df)

    # remove stop words
    stop_words = StopWordsRemover(inputCol="words", outputCol="clear_text")
    df = stop_words.transform(df)

    # remove duplicate words
    df = df.withColumn("clear_text", F.concat_ws(" ", F.array_distinct(F.col("clear_text"))))

    # convert text to lowercase
    df = df.withColumn("clear_text", lower(col("clear_text")))
    return df

def get_tweets_by_date(df, start_date, end_date):
    # filter the dataframe by the given date range
    # Create a UDF to filter the dataframe by the given date range
    if df.count == 0 :
        return df
    start_date = datetime.strptime(start_date, '%m/%d/%Y')
    end_date = datetime.strptime(end_date, '%m/%d/%Y')
    date_filter = udf(lambda x: start_date <= datetime.strptime(x, '%m/%d/%Y') <= end_date, BooleanType())
    filtered_df = df.filter(date_filter(col("day")))
    filtered_df.cache() # cache this to be run faster on the second time
    return filtered_df

def count_tweets(df, kw_list):
    # Filter tweets based on the keywords input
    query = "("
    for set_ in kw_list:
        query += "("
        for keyword in set_:
            query += f"instr(clear_text, '{keyword.lower()}') > 0 OR "
        query = query[:-3] + ") AND "
    query = query[:-4] + ")"

    filtered_df = df.filter(expr(query))
    # filtered_df.cache()
    return filtered_df, filtered_df.count()


def export_df(file, counts, total, dates):
    df = pd.DataFrame({'Filename': pd.Series(dtype='str'),
                   'Country': pd.Series(dtype='str'),
                   'Counts': pd.Series(dtype='int'),
                   'Total': pd.Series(dtype='int'),
                   'Percentage': pd.Series(dtype='float'),
                   'Start Date': pd.Series(dtype='str'),
                   'End Date': pd.Series(dtype='str')})

    for country,count in counts.items():
        percentage = (count/total)*100 if total else 0
        entry = pd.DataFrame.from_dict([{
            'Filename': file,
            'Country':  country,
            'Counts': count,
            'Total': total,
            'Percentage':  percentage,
            'Start Date': str(dates[0]),
            'End Date': str(dates[1])}])

        df = pd.concat([df, entry], ignore_index=True)
    return df

def get_results(df, start_date, end_date, winner_kws, worldCup_kws, countries):
    filtered_df = get_tweets_by_date(df, start_date, end_date)
    df_filt, _count = count_tweets(filtered_df, [ winner_kws, worldCup_kws])
    results = {}

    for i, country in enumerate(countries):
        df_filt, count = count_tweets(df_filt, [country])
        results[country[0]] = count if count else 0
        # Combining the dataframes
        if i > 0 :
            combined_df = combined_df.union(df_filt)
        else:
             combined_df = df_filt
    # Counting the number of unique rows
    # num_unique_rows = combined_df.distinct().count()
    total_combined = combined_df.count()

    return results, total_combined


df = df_preproc(df_init)

# Filter keywords
winner_kws = ["Win", "winner", "lift", "Champion", "champions" "trophy", "winning", "won"]
worldCup_kws = ["WolrdCup", "WC", "WorldCup", "FifaWorldCup", "Fifa", "Cup", "Trophy", "Champion", "Fifa22", "Fifa2022", "WolrdCup2022", "FifaWorldCup2022","QatarWolrdCup2022"]

## First Experiment ##
country_kw1 = ['Argentina', 'AR', 'ARG', 'Albiceleste']
country_kw2 = ['France', 'FR', 'FRA', 'Bleus']
country_kw3 = ['Marocco', 'maroco','MAR', 'MA','Atlas']
country_kw4 = ['Croatia', 'HR', 'CRO', 'Vatreni']
countries1 = [country_kw1,country_kw2,country_kw3,country_kw4]

start_dates = ["11/20/2022","12/02/2022","12/07/2022","12/11/2022"]
end_dates = ["11/21/2022","12/02/2022","12/08/2022","12/12/2022"]

for n in range(len(start_dates)):
    results, total_combined = get_results(df, start_dates[n], end_dates[n], winner_kws, worldCup_kws,countries1)
    res_df = export_df(inputUri1.split('/')[-1], results, total_combined, [start_dates[n], end_dates[n]])
    if n == 0:
        final_df = res_df
    else:
        final_df = pd.concat([final_df, res_df],ignore_index=True)

if "gs://" in outputUri:
    final_df.to_csv(outputUri + f"results_2022.csv")
else:
    Path("output").mkdir(parents=True, exist_ok=True)
    final_df.to_csv("output/" + f"results_2022.csv")

# Stop the SparkContext
sc.stop()

## Second Experiment ##
# Create a Spark configuration and context
conf = SparkConf().setAppName("WorldCupTweets").setMaster('local[4]')
# conf = SparkConf().setAppName("WorldCupTweets").setMaster('local[4]')
sc = SparkContext(conf=conf)
# Create a SparkSession
spark = SparkSession.builder.appName("WorldCupTweets").getOrCreate()

# Read the CSV file as a DataFrame
df_init = pd.read_csv(inputUri2,
            usecols = ['date', 'user_description', 'text', 'hashtags',
                        'user_name','user_location'])
df_init=df_init.reindex(columns=['user_name','user_location','user_description','date','text','hashtags'])

# remove null dates
df_init = df_init[~df_init['date'].isnull()]
tweets_df = spark.createDataFrame(df_init,df_schema)
df = df_preproc(df_init)

# Filter keywords
# Filter keywords
winner_kws = ["Win", "winner", "lift", "Champion", "champions" "trophy", "winning", "won"]
worldCup_kws = ["WolrdCup", "WC", "WorldCup", "FifaWorldCup", "Fifa", "RUssia2018","Cup", "Trophy", "Champion", "Fifa18", "Fifa2018", "WolrdCup2018", "FifaWorldCup2018", "RussiaWolrdCup2018"]

country_kw1 = ['England', 'ENG', 'EN', 'LIONS']
country_kw2 = ['France', 'FR', 'FRA', 'Bleus']
country_kw3 = ['Croatia', 'HR', 'CRO', 'Vatreni']
country_kw3 = ['Belgium', 'BE','BELG',  'BEL', 'Devils', 'Red', 'Reds']
countries1 = [country_kw1,country_kw2,country_kw3,country_kw4]

start_dates = ["06/29/2018","07/01/2018","07/02/2018","07/10/2018"]
end_dates = ["06/30/2018","07/01/2018","07/04/2018","07/11/2018"]

for n in range(len(start_dates)):
    results, total_combined = get_results(df, start_dates[n], end_dates[n], winner_kws, worldCup_kws,countries1)
    res_df = export_df(inputUri2.split('/')[-1], results, total_combined, [start_dates[n], end_dates[n]])
    if n == 0:
        final_df = res_df
    else:
        final_df = pd.concat([final_df, res_df],ignore_index=True)


if "gs://" in outputUri:
    final_df.to_csv(outputUri + f"results_2018.csv")
else:
    Path("output").mkdir(parents=True, exist_ok=True)
    final_df.to_csv("output/" + f"results_2018.csv")
