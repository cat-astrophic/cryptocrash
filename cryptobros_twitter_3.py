# This script uses twint to harvest relevant twitter account data

# Importing required modules

import twint
import nest_asyncio
import pandas as pd
import numpy as np

# Where to write the tweets data

username = ''
writeto = 'C:/Users/' + username + '/Documents/Data/cryptobros/data/'

# Read in the accounts data

accounts_df = pd.read_csv('C:/Users/' + username + '/Documents/Data/cryptobros/data/accounts.csv')

# Randomly subset control accounts to match the treated groups in size

l = len(accounts_df[accounts_df.Type == 'control'])
lx = len(accounts_df) - l
tdf = accounts_df[accounts_df.Type != 'control'].reset_index(drop = True)
cdf = accounts_df[accounts_df.Type == 'control'].reset_index(drop = True)
keep = np.random.choice(l,lx,replace = False)
cdf = cdf[cdf.index.isin(keep)]
df = pd.concat([tdf,cdf], axis = 0).reset_index(drop = True)

# Use nest_asyncio to permit asynchronous loops

nest_asyncio.apply()

# Initializing the main dataframes

pre_df = pd.DataFrame()
may_df = pd.DataFrame()
post_df = pd.DataFrame()

# Using twint to get pre-event data

for u in list(df.Username):
    
    try:
        
        print(u + ' :: Account ' + str(list(df.Username).index(u)) + ' of ' + str(len(df.Username)) + '.......')
        t = twint.Config()
        t.Username = u
        t.Since = '2022-01-01'
        t.Until = '2022-04-30'
        t.Lang = 'en'
        t.Store_csv = True
        t.Pandas = True
        twint.run.Search(t)
        twint.storage.panda.save
        Tweets_df = twint.storage.panda.Tweets_df
        pre_df = pd.concat([pre_df, Tweets_df], axis = 0)
        
    except:
        
        continue

# Using twint to get may-event data

for u in list(df.Username):
    
    try:
        
        print(u + ' :: Account ' + str(list(df.Username).index(u)) + ' of ' + str(len(df.Username)) + '.......')
        t = twint.Config()
        t.Username = u
        t.Since = '2022-05-01'
        t.Until = '2022-05-31'
        t.Lang = 'en'
        t.Store_csv = True
        t.Pandas = True
        twint.run.Search(t)
        twint.storage.panda.save
        Tweets_df = twint.storage.panda.Tweets_df
        may_df = pd.concat([may_df, Tweets_df], axis = 0)
        
    except:
        
        continue

# Using twint to get post-event data

for u in list(df.Username):
    
    try:
        
        print(u + ' :: Account ' + str(list(df.Username).index(u)) + ' of ' + str(len(df.Username)) + '.......')
        t = twint.Config()
        t.Username = u
        t.Since = '2022-06-01'
        t.Until = '2022-08-31'
        t.Lang = 'en'
        t.Store_csv = True
        t.Pandas = True
        twint.run.Search(t)
        twint.storage.panda.save
        Tweets_df = twint.storage.panda.Tweets_df
        post_df = pd.concat([post_df, Tweets_df], axis = 0)
        
    except:
        
        continue

# Saving the tweets

pre_df.to_csv(writeto + 'tweets_pre.csv', index = False)
may_df.to_csv(writeto + 'tweets_may.csv', index = False)
post_df.to_csv(writeto + 'tweets_post.csv', index = False)

