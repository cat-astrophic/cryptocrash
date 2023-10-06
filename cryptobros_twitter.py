# This script uses twint to harvest relevant twitter data

# Importing required modules

import twint
import nest_asyncio
import pandas as pd

# Where to write the tweets data

writeto = 'D:/cryptobros/data/'

# Use nest_asyncio to permit asynchronous loops

nest_asyncio.apply()

# Keywords to search for

keywords = ['wagmi', 'investing']

# Initializing the main dataframe

twitter_df = pd.DataFrame()

# Using twint to get data

for k in keywords:
    
    t = twint.Config()
    t.Search = k
    t.Since = '2022-01-01'
    t.Until = '2022-08-31'
    t.Lang = 'en'
    t.Store_csv = True
    t.Pandas = True
    twint.run.Search(t)
    twint.storage.panda.save
    Tweets_df = twint.storage.panda.Tweets_df
    twitter_df = pd.concat([twitter_df, Tweets_df], axis = 0)

# Removing any duplicate tweets

twitter_df = twitter_df[twitter_df].reset_index(drop = True)

# Writing the complete raw data file

twitter_df.to_csv(writeto + 'raw_twitter_data.csv', index = False)

