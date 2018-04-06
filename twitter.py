# 한글 70 140 lang = 'ko'
from twitterscraper import query_tweets
import datetime as dt

if __name__ == '__main__':
    list_of_tweets = query_tweets("KOSPI OR KOSDAQ", 10)

    #print the retrieved tweets to the screen:
    for tweet in query_tweets("KOSPI OR KOSDAQ ", 10,
                                  begindate=dt.date(2018,3,1),
                                  enddate=dt.date(2018,4,1)):
        print(tweet.timestamp)
        print(tweet.text)

    #Or save the retrieved tweets to file:
    file = open('output.txt','bw')
    for tweet in query_tweets("KOSPI OR KOSDAQ", 10):
        file.write(str(tweet.timestamp).encode('utf-8'))
        file.write(tweet.text.encode('utf-8'))
    file.close()