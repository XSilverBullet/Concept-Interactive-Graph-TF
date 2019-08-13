import networkx as nx

# G = nx.Graph()
# # G.add_edges_from([(1,2,dict(weight=[1,2,3])),
# #                    (1,2,dict(height=1))])
#
# G.add_node(1, color='blue')
# G.add_node(2, size=12,)
# #G.add_node(1, male=1)
# # for n, nbrs in FG.adjacency():
# #     for nbr, eattr in nbrs.items():
# #         data1 = eattr['weight']
# #         data2 = eattr['height']
# #         print(n,nbr,data1, data2)
# #
#
# for v in G.nodes():
#      print(G.node[v])


# from networkx.algorithms import community
# G = nx.barbell_graph(5, 1)
# communities_generator = community.girvan_newman(G)
# top_level_communities = next(communities_generator)
# next_level_communities = next(communities_generator)

# import torch
# import numpy
# a = numpy.array([[1, 2, 3],
#                  [1, 3, 4]])
# t = torch.from_numpy(a)
# print(t)

import pandas as pd

df = pd.read_csv("../../../../data/raw/event-story-cluster/same_event_doc_pair.txt", sep='|')

cnt=0
for i in range(df.shape[0]):

    text1 = ''.join(df['content1'][0].split(' '))
    text2 = ''.join(df['content2'][0].split(' '))

    if len(text1) <350 or len(text2) < 350:
        cnt = cnt + 1

print(cnt)


