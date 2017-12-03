import tweepy
from tweepy import OAuthHandler
import json
 
consumer_key = 'YOUR-CONSUMER-KEY'
consumer_secret = 'YOUR-CONSUMER-SECRET'
access_token = 'YOUR-ACCESS-TOKEN'
access_secret = 'YOUR-ACCESS-SECRET'
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth)

# Currently gives us a dump of raw meta data that can be further processed
def process_or_store(tweet):
    print(json.dumps(tweet))

for status in tweepy.Cursor(api.home_timeline).items(10):
    data = json.dumps(status._json)

# # Dumps text from first 10 statuses to interpreter
#  for status in tweepy.Cursor(api.home_timeline).items(10):
#     # Process a single status
#     print(status.text)

# Prints to a JSON file
with open('twitter.json', 'w') as outfile:
    json.dump(data, outfile)