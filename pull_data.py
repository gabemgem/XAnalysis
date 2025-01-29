import tweepy
import pandas as pd
import json
import pickle as pkl

# # Twitter API credentials
# bearer_token = "AAAAAAAAAAAAAAAAAAAAAEeQyQEAAAAAz7RyNP5HHBjnPkdJap%2Fb%2Bna78Dc%3D1czDzE57XZmVW65ezR9d1lGIfIRZQwyQ6EUL0xZYSQy80AZiES"
#
# # Create API client
# client = tweepy.Client(bearer_token)
#
# # df = pd.read_csv('data/notes-00000.tsv', sep='\t')
# # print(df['tweetId'].head())
# #
# # tweetid = df['tweetId'][0]
# # print(tweetid)
# tweetid = '1783159712986382830'
#
# # Get tweet by ID
# tweet = client.get_tweet(tweetid)
#
# # save tweet data
# with open('tweet.pkl', 'wb') as f:
#     pkl.dump(tweet, f)

import sys
sys.setrecursionlimit(5000)
# load tweet data
with open('tweet.pkl', 'rb') as f:
    tweet = pkl.load(f)

print(tweet.data)
