"""
cluster.py
"""

# Imports
import copy
import networkx as nx
import pickle


def girvan_newman(graph):
    G=graph.copy()
    
    if G.order() == 1:
        return [sorted(G.nodes())]
        
    components = [c for c in nx.connected_component_subgraphs(G)]
    eb = nx.edge_betweenness_centrality(G)
    edge_to_remove=sorted(eb.items(), key=lambda x: x[1], reverse=True)
    initial_num_components = len(components)
    
    while len(components) == initial_num_components:
        G.remove_edge(*edge_to_remove[0][0])
        del edge_to_remove[0]
        components = [c for c in nx.connected_component_subgraphs(G)]
   
    result = [c for c in components]
    
    return result


def main():

    with open('graph.pickle', 'rb') as f:
        graph = pickle.load(f)  
    
    clusters = girvan_newman(graph)
    
    with open('clusters.pickle', 'wb') as f:
        pickle.dump(clusters, f, pickle.HIGHEST_PROTOCOL)
        

    
    
if __name__ == '__main__':
    main()