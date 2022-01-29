import pandas as pd
import requests
import pandas as pd
# For parsing the dates received from twitter in readable formats
from datetime import date, datetime, timedelta
import pytz
#To add wait time between requests
import time
import logging

logger = logging.getLogger(__name__)
class TwitterExtractor:
    
    def __init__(self, companies, api_token, actual_datetime, hours=24, max_tweets_search=100, results_by_page=100):
        self._companies = companies
        self._api_token = api_token
        self._hours = hours
        self._max_tweets_search = max_tweets_search
        self._results_by_page = results_by_page
        self._actual_datetime = actual_datetime

    def _create_headers(self):
        headers = {"Authorization": "Bearer {}".format(self._api_token)}
        return headers
    
    def _create_url(self, keyword, start_date, max_results = 10):
    
        search_url = "https://api.twitter.com/2/tweets/search/recent" #Change to the endpoint you want to collect data from

        #change params based on the endpoint you are using
        query_params = {'query': keyword,
                        'start_time': start_date,
                        'max_results': max_results,
                        'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
                        'tweet.fields': 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source',
                        'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
                        'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                        'next_token': {}}
        return (search_url, query_params)
    
    def _connect_to_endpoint(self, url, headers, params, next_token = None):
        params['next_token'] = next_token   #params object received from create_url function
        response = requests.request("GET", url, headers = headers, params = params)
        #print("Endpoint Response Code: " + str(response.status_code))
        if response.status_code != 200:
            raise Exception(response.status_code, response.text)
        return response.json()
    
    def _extract_tweets(self, company, max_tweets=1000, results_by_page=50):

        start_time = (self._actual_datetime - timedelta(hours=self._hours)).replace(tzinfo=pytz.timezone('America/Sao_Paulo'))
        start_time =start_time.isoformat()
        headers = self._create_headers()
        keyword = f"{company['query']} lang:pt"

        search = True
        next_token = None
        df_tweets = pd.DataFrame()
        page = 1
        while search and df_tweets.shape[0] < max_tweets:
            url = self._create_url(keyword, start_time, results_by_page)
            json_response = self._connect_to_endpoint(url[0], headers, url[1], next_token=next_token)


            for tweet in json_response['data']:
                tweet_info = {'company':company['name'], 'replied_to': False, 'quoted': False}
                retweet = False
                if 'referenced_tweets' in tweet: 
                    for ref in tweet['referenced_tweets']:
                        if ref['type'] == 'retweeted':
                            retweet = True
                        else:
                            tweet_info[ref['type']] = True
                if retweet:
                    continue
                for metric in tweet['public_metrics']:
                    tweet_info[metric] = tweet['public_metrics'][metric]
                tweet_info['text'] = tweet['text']
                tweet_info['source'] = tweet['source']
                tweet_info['lang'] = tweet['lang']
                tweet_info['reply_settings'] = tweet['reply_settings']
                tweet_info['created_at'] = tweet['created_at']
                df_tweets = df_tweets.append(tweet_info, ignore_index=True)
            if 'next_token' in json_response['meta']:
                next_token = json_response['meta']['next_token']
            else:
                next_token = None
            print(f"\rpage: {page}, tweets: {df_tweets.shape[0]}", end='')
            page += 1
            if next_token is None:
                search = False
        print()
        
        return df_tweets
    
    def extract_last_tweets(self):
        new_tweets = pd.DataFrame()

        for i, (company_id, company) in enumerate(self._companies.items()):
            logger.info(f"[{i+1}/{len(self._companies)}] Extracting tweets about company {company['name']}...")
            extracted_tweets = self._extract_tweets(company, 
                                                    max_tweets=self._max_tweets_search, 
                                                    results_by_page=self._results_by_page)
            new_tweets = new_tweets.append(extracted_tweets, ignore_index=True)
            time.sleep(30)

        new_tweets = new_tweets[new_tweets.lang == 'pt']
        new_tweets['created_at'] = pd.to_datetime(new_tweets['created_at'])
        companies_names = [c["name"] for _, c in self._companies.items()] 
        new_tweets = new_tweets[new_tweets.company.isin(companies_names)]
        logger.info(f'Extraction of {new_tweets.shape[0]} tweets completed successfully!')
        return new_tweets
        