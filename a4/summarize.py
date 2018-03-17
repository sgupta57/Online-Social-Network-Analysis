"""
summarize.py
"""

# Imports
import pickle

def read_data():
    with open('graph.pickle', 'rb') as f:
        graph = pickle.load(f) 
    
    print('\nNumber of users collected: %d' % len(graph.nodes()))
    
    with open('tweets.pickle', 'rb') as f:
        tweets = pickle.load(f)
        
    print('\nNumber of messages collected: %d' % len(tweets))
    
    with open('clusters.pickle', 'rb') as f:
        clusters = pickle.load(f)
        
    print('\nNumber of communities discovered: %d' %(len(clusters)))
    
    total=0
    for i in range(len(clusters)):
        total += (clusters[i].order())
    average =total/(len(clusters))
    
    print('\nAverage number of users per community: %f' %average)
    
    with open('classify.pickle', 'rb') as f:
        (positives,negatives,mixed) = pickle.load(f)
        
    print('\nNumber of instances per class found: %d positive instances , %d negative instances and %d mixed instances' 
    %(len(positives), len(negatives), len(mixed)))

    print('\nOne example from each class:')
    for tweet, pos, neg in sorted(positives, key=lambda x: x[1], reverse=False):
        positive=(pos,neg,tweet)
    print('Example of positive class:')
    print(positive)
    
    for tweet, pos, neg in sorted(negatives, key=lambda x: x[2], reverse=False):
        negative=(neg,pos,tweet)
    print('Example of negative class:')
    print(negative)
    
    for tweet, pos, neg in sorted(mixed, key=lambda x: x[2], reverse=True):
        mixed=(pos,neg,tweet)
    print('Example of mixed class:')
    print(mixed)
    
    file = open("summary.txt", "w")
    file.write(('Number of users collected: %d' % len(graph.nodes())))
    file.write('\nNumber of messages collected: %d' % len(tweets))
    file.write('\nNumber of communities discovered: %d' %(len(clusters)))
    file.write('\nAverage number of users per community: %f' %average)
    file.write('\nNumber of instances per class found: %d positive instances , %d negative instances and %d mixed instances' 
    %(len(positives), len(negatives), len(mixed)))
    file.write('\nOne example from each class: ')
    file.write('\nExample of positive class:')
    file.write(positive[2])
    file.write('\nExample of negative class:')
    file.write(negative[2])
    file.write('\nExample of mixed class:')
    file.write(mixed[2])
    file.close
    
def main():
    read_data()
    
    
    
if __name__ == '__main__':
    main()