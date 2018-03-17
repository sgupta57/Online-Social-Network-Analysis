"""
collect.py
"""

# Imports
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
import pickle

consumer_key = 'dVGkQAqE1Tvqj4SIFIiCeH0ez'
consumer_secret = 'nJ2bcP0Pg2BTS1xxB8tBJxVWGnIWuYnCrnNmIXpdT3HVEwFb9M'
access_token = '768597871536984065-be1ZlhBsfDQQ3roZ8y1GgDtL3ksTWGj'
access_token_secret = 'FLCXGEpEYCHb9rEzv2dLJmwAOVSNPDORgdCZCqlInFgGq'


def get_twitter():
    
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def read_screen_names(filename):
    
    screen_names=open(filename,'r')
    
    return ([s.strip() for s in screen_names.readlines()])


def robust_request(twitter, resource, params, max_tries=5):
    
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)

def get_users(twitter, screen_names):
    
    request= []
    request= robust_request(twitter, 'users/lookup', {'screen_name':screen_names})
    users= [u for u in request]
    
    return (users)
           
def get_friends(twitter, screen_name):
    
    request= []
    request= robust_request(twitter, 'followers/ids',{'screen_name':screen_name,'count':500})
    friends= [f for f in request]
    friends.sort()
    
    return friends

def add_all_friends(twitter, users):
    
    for i in range(len(users)):
        users[i]['friends'] = get_friends(twitter, users[i]['screen_name']) 
        
def create_graph(users):
    
    graph=nx.Graph()

    for i in range(len(users)):
       user=(users[i]['screen_name'])
       graph.add_node(user)
       for f in (users[i]['friends']):
           graph.add_edge(users[i]['screen_name'],str(f))
    
    return graph
    
def draw_network(graph, users, filename):
    
    candidates=set(u['screen_name'] for u in users)
    labels={n: n if n in candidates else '' for n in graph.nodes()}
    
    plt.figure(figsize=(12,12))
    
    nx.draw_networkx(graph,alpha=.5, labels=labels, width=.1, node_size=100)

    plt.axis("off")
    plt.savefig(filename)

def get_tweets(twitter, screen_name):
    
    tweets = []
    
    for s in screen_name:
        request=robust_request(twitter,'search/tweets', {'q': s, 'lang':'en', 'count': 100})
        for t in request:
            tweets.append(t)
    
    return tweets 
    
def main():
    twitter = get_twitter()
    screen_names = read_screen_names('candidates.txt')
    
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    
    add_all_friends(twitter, users)
    graph = create_graph(users)
    draw_network(graph, users, 'network.png')
    
    with open('graph.pickle', 'wb') as f:
        pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)
        
    tweets = get_tweets(twitter,screen_names)
    
    with open('tweets.pickle', 'wb') as f:
        pickle.dump(tweets, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
