# This script classifies twitter accounts

# Importing required modules

import pandas as pd

# Where to write the tweets data

username = ''
writeto = 'C:/Users/' + username + '/Documents/Data/cryptobros/data/'

# Read in the data

twitter_df = pd.read_csv('C:/Users/' + username + '/Documents/Data/cryptobros/data/raw_twitter_data.csv')

# Subsetting the data based on potential treated or control status

wagmi_df = twitter_df[twitter_df.search == 'wagmi'].reset_index(drop = True)
control_df = twitter_df[~twitter_df.id.isin(list(wagmi_df.id))].reset_index(drop = True)

wagmi_user_ids = list(wagmi_df.user_id.unique())
wagmi_usernames = list(wagmi_df.username.unique())
wagmi_names = list(wagmi_df.name.unique())

eth_ids = [x for x in range(len(wagmi_names)) if '.eth' in wagmi_names[x]]
eth_user_ids = [wagmi_user_ids[x] for x in eth_ids]
eth_usernames = [wagmi_usernames[x] for x in eth_ids]
eth_names = [wagmi_names[x] for x in eth_ids]

control_user_ids = list(control_df.user_id.unique())
control_usernames = list(control_df.username.unique())
control_names = list(control_df.name.unique())

# Building a new df and saving to file

user_ids = wagmi_user_ids + eth_user_ids + control_user_ids
usernames = wagmi_usernames + eth_usernames + control_usernames
cats = ['wagmi']*len(wagmi_usernames) + ['eth']*len(eth_usernames) + ['control']*len(control_usernames)

user_ids = pd.Series(user_ids, name = 'UserID')
usernames = pd.Series(usernames, name = 'Username')
cats = pd.Series(cats, name = 'Type')
accounts_df = pd.concat([user_ids, usernames, cats], axis = 1)
accounts_df = accounts_df.drop_duplicates(['UserID'])
accounts_df.to_csv(writeto + 'accounts.csv', index = False)

