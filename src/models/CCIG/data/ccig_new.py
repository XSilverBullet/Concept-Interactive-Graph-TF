from config import *
# from girvan_newman import *
from sentence_pair_score import *
from resource_loader import *
from util.nlp_utils import split_sentence
from util.list_utils import common, substract, remove_values_from_list
import networkx as nx
import matplotlib.pyplot as plt

LANGUAGE = "Chinese"
#
# IDF = load_IDF("event_story")
# STOPWORDS = load_stopwords(LANGUAGE)
EMPTY_VERTEX_NAME = ""
TITLE_VERTEX_NAME = "_TITLE_"

ALPHA = 0.1  # sentence score paragraph index decay parameter
BETA = 0.3  # sentence score sentence index decay parameter


def print_ccig(g, sentences1, sentences2):
    """
     Print ccig.
     :param g: concept community interaction graph.
     :param sentences1: sentences from document 1.
     :param sentences2: sentences from document 2.
     """
    # g = nx.Graph()
    print('\n<graph>')
    for v in g.nodes():
        idx1 = list(set(g.node[v]['sentidxs1']))
        idx2 = list(set(g.node[v]['sentidxs2']))
        text1 = [sentences1[i] for i in idx1]
        text1 = '\n'.join(text1)
        text2 = [sentences2[i] for i in idx2]
        text2 = '\n'.join(text2)
        print("<vertex>" + str(v))
        print("Keywords: " + ",".join(g.node[v]["concepts"]))
        print("Sentences 1: ")
        print(text1)
        print("Sentences 2: ")
        print(text2)
        print("</vertex>" + str(v))

    for e in g.edges():
        weight = g[e[0]][e[1]]["weight_tfidf"]
        concepts = g[e[0]][e[1]]["concepts"]
        concepts = ",".join(concepts)
        print("Edge (" + str(e[0]) + ", " + str(e[1]) + ")")
        print("Keywords: " + concepts)
        print("Weight: " + str(weight))
    print("num_v_L: " + str(g.graph["num_v_L"]))
    print("num_v_R: " + str(g.graph["num_v_R"]))
    print("num_v_LR: " + str(g.graph["num_v_LR"]))
    print("match rate: " + str(g.graph["num_v_LR"] / g.number_of_nodes()))
    print("</graph>\n")


def draw_ccig(g):
    nx.draw(g)
    plt.show()


def construct_ccig(sentences, concepts, title=None, use_cd=True, betweenness_threshold_coef=1.0, max_c_size=10,
                   min_c_size=3, IDF=None):
    """
     Given a segmented text and a list of concepts,
     construct concept community interaction graph.
     :param sentences: a list of sentences.
     :param concepts: a list of concepts.
     :return: a concept community interaction graph.
     """
    g = nx.Graph()

    concepts = list(set(concepts))
    concepts = remove_values_from_list(concepts, EMPTY_VERTEX_NAME)

    if len(sentences) == 0 or len(concepts) == 0:
        print("No concept in concepts list.")
        return None
    if len(concepts) > 70:
        print("Too many concepts.")
        return None

    # get concept communities
    if use_cd:
        concept_communities = get_concept_communities(sentences, concepts, betweenness_threshold_coef, max_c_size,
                                                      min_c_size)

    else:
        concept_communities = [[c] for c in concepts]

    if use_cd:
        cname_sentidxs = assign_sentences_to_concept_communities(
            sentences, concept_communities, IDF)
    else:
        cname_sentidxs = assign_sentences_to_concepts(sentences, concepts)

    # initialize vertex properties
    concept_vertexidxs_map = {}

    for c in concepts:
        concept_vertexidxs_map[c] = []

    g.add_node(0, name=EMPTY_VERTEX_NAME, concepts=[], sentidxs=cname_sentidxs[EMPTY_VERTEX_NAME])
    # g.add_node(0)
    # g.node[0]['name'] = EMPTY_VERTEX_NAME
    # g.node[0]['concepts'] = []
    # g.node[0]['sentidxs'] = cname_sentidxs[EMPTY_VERTEX_NAME]

    # print(g.node[0])
    i = 1

    for community in concept_communities:
        cname = community2name(community)

        if len(cname_sentidxs[cname]) == 0:
            continue

        g.add_node(i, name=cname, concepts=community, sentidxs=cname_sentidxs[cname])

        for concept in community:
            concept_vertexidxs_map[concept].append(i)
        i = i + 1

    # edges by connective entences
    # dic
    eprop_name = {}
    eprop_concepts = {}
    eprop_sentidxs = {}
    eprop_weight_numsent = {}
    eprop_weight_tfidf = {}

    for sent_idx in range(len(sentences)):
        sent = sentences[sent_idx]
        words = str(sent).split()
        intersect = set(words).intersection(set(concepts))

        if len(intersect) == 0:
            continue

        related_vertexidxs = []

        for c in intersect:
            related_vertexidxs.extend(concept_vertexidxs_map[c])
        related_vertexidxs = list(set(related_vertexidxs))

        # print("related_vertex_idx:")
        # print(related_vertexidxs)

        num_related_v = len(related_vertexidxs)

        if num_related_v < 2:
            continue

        for j in range(num_related_v):
            v1_idx = related_vertexidxs[j]
            for k in range(j, num_related_v):
                if j == k:
                    continue
                v2_idx = related_vertexidxs[k]

                source_idx = min(v1_idx, v2_idx)
                target_idx = max(v1_idx, v2_idx)

                e = (source_idx, target_idx)
                if not g.has_edge(source_idx, target_idx):
                    # g.add_edge(source_idx, target_idx)

                    eprop_sentidxs[e] = [sent_idx]
                    eprop_concepts[e] = list(intersect)

                    g.add_edge(source_idx, target_idx)

                    # g.add_edges_from([(source_idx, target_idx, dict(sentidxs=eprop_sentidxs[e])),
                    #                   (source_idx, target_idx, dict(concepts=eprop_concepts[e]))])

                else:
                    old_idxs = list(eprop_sentidxs[e])
                    old_idxs.append(sent_idx)
                    eprop_sentidxs[e] = old_idxs

                    old_concepts = list(eprop_concepts[e])
                    old_concepts.extend(intersect)
                    eprop_concepts[e] = list(set(old_concepts))

                g[source_idx][target_idx]['sentidxs'] = eprop_sentidxs[e]
                g[source_idx][target_idx]['concepts'] = eprop_concepts[e]

    # assign vertex names and weights
    for e in g.edges():
        eprop_name[e] = " ".join(eprop_concepts[e])
        eprop_weight_numsent[e] = float(len(eprop_sentidxs[e]))
        eprop_weight_tfidf[e] = 0.0

        g[e[0]][e[1]]['weight_numsent'] = eprop_weight_numsent[e]
        g[e[0]][e[1]]['weight_tfidf'] = eprop_weight_tfidf[e]

    # edges by node text similarity
    WEIGHT_THRESHOLD = 0.001  # NOTICE: smaller threshold leads to more edges

    numv = g.number_of_nodes()

    for i in range(numv):
        for j in range(i, numv):
            if j == i:
                continue
            v1 = g.node[i]
            v2 = g.node[j]
            idxs1 = list(set(v1['sentidxs']))
            idxs2 = list(set(v2['sentidxs']))

            text1 = [sentences[s] for s in idxs1]
            text1 = " ".join(text1)
            text2 = [sentences[s] for s in idxs2]
            text2 = " ".join(text2)

            w = tfidf_cos_sim(text1, text2, IDF)

            if w >= WEIGHT_THRESHOLD:
                e = (i, j)
                if not g.has_edge(i, j):
                    eprop_sentidxs[e] = []
                    eprop_concepts[e] = []
                    eprop_weight_numsent[e] = 0.0
                    eprop_name[e] = ""
                    g.add_edges_from([
                        (i, j, dict(sentidxs=eprop_sentidxs[e])),
                        (i, j, dict(concepts=eprop_concepts[e])),
                        (i, j, dict(weight_numsent=eprop_weight_numsent[e])),
                        (i, j, dict(weight_name=eprop_name[e]))
                    ])
                eprop_weight_tfidf[e] = w
                g[i][j]['weight_tfidf'] = eprop_weight_tfidf[e]
    if title is not None:
        g.add_nodes_from('TITLE', name=TITLE_VERTEX_NAME, sentidxs=[], concepts=[])

    #g.add_nodes_from('T', name=TITLE_VERTEX_NAME, sentidxs=[], concepts=[])
    # calculate vertex scores
    pr = nx.pagerank(g, weight='weight_tfidf')
    bt = nx.betweenness_centrality(g, weight='weight_tfidf')
    #print(bt)
    try:
        katz = nx.katz_centrality(g, weight='weight_tfidf')
    except:
        katz = [0.0 for i in range(numv)]
    #numv = len(pr)
    for i in g.nodes():
        #print(i)
        g.node[i]['pagerank'] = pr[i]
        g.node[i]['betweenness'] = bt[i]
        g.node[i]['katz'] = katz[i]

    ebt = nx.edge_betweenness(g, weight='weight_tfidf')
    #print(ebt)
    #print(g.nodes())
    for i in range(len(g.nodes())):
        for j in range(i, len(g.nodes())):
            if j == i:
                continue
            if g.has_edge(i, j):
                g[i][j]['betweenness'] = ebt[(i, j)]

    return g

    #
    # nx.draw(g, with_labels=True)
    # plt.show()
    # for e in g.edges():

    # nx.draw(g)
    # plt.show()
    # print(g)

    # vprop_name[v_empty] = EMPTY_VERTEX_NAME
    # vprop_concepts[v_empty] = []
    # vprop_sentidxs[v_empty] = cname_sentidxs[EMPTY_VERTEX_NAME]


def construct_align_ccig(sentences1, sentences2, concepts1, concepts2,
                         title1=None, title2=None, use_cd=True,
                         betweenness_threshold_coef=1.0, max_c_size=10, min_c_size=3, IDF=None):
    num_sentences1 = len(sentences1)
    sentences = list(sentences1)
    sentences.extend(sentences2)
    concepts = list(concepts1)
    concepts.extend(concepts2)
    g = construct_ccig(sentences, concepts, title1, use_cd, betweenness_threshold_coef,
                       max_c_size, min_c_size, IDF)
    if g is None:
        return None

    # vertices
    vprop_exist = {}  # "L" or "R" or "LR"
    vprop_sentidxs1 = {}
    vprop_sentidxs2 = {}

    lr = 0
    l = 0
    r = 0
    for v in g.nodes():
        sentidxs = list(g.node[v]['sentidxs'])
        sentidxs1 = [x for x in sentidxs if x < num_sentences1]
        sentidxs2 = [x - num_sentences1 for x in sentidxs
                     if x >= num_sentences1]
        vprop_sentidxs1[v] = sentidxs1
        vprop_sentidxs2[v] = sentidxs2

        if len(sentidxs1) > 0 and len(sentences2) > 0:
            vprop_exist[v] = 'LR'
            lr = lr + 1
        elif len(sentidxs1) > 0:
            vprop_exist[v] = 'L'
            l = l + 1
        elif len(sentidxs2) > 0:
            vprop_exist[v] = 'R'
            r = r + 1
        else:
            vprop_exist[v] = 'NULL'

        g.node[v]['exist'] = vprop_exist[v]
        g.node[v]['sentidxs1'] = vprop_sentidxs1[v]
        g.node[v]['sentidxs2'] = vprop_sentidxs2[v]

    eprop_exist = {}
    eprop_sentidxs1 = {}
    eprop_sentidxs2 = {}
    for e in g.edges():

        sentidxs = list(g[e[0]][e[1]]["sentidxs"])
        sentidxs1 = [x for x in sentidxs if x < num_sentences1]
        sentidxs2 = [x - num_sentences1 for x in sentidxs if x >= num_sentences1]

        eprop_sentidxs1[e] = sentidxs1
        eprop_sentidxs2[e] = sentidxs2
        if len(sentidxs1) > 0 and len(sentidxs2) > 0:
            eprop_exist[e] = "LR"
        elif len(sentidxs1) > 0:
            eprop_exist[e] = "L"
        elif len(sentidxs2) > 0:
            eprop_exist[e] = "R"
        else:
            eprop_exist[e] = "NULL"
        g[e[0]][e[1]]['exist'] = eprop_exist[e]
        g[e[0]][e[1]]['sentidxs1'] = eprop_sentidxs1[e]
        g[e[0]][e[1]]['sentidxs2'] = eprop_sentidxs2[e]

    g.graph['num_v_L'] = l
    g.graph['num_v_R'] = r
    g.graph['num_v_LR'] = lr

    return g


def text_pair2ccig(text1, text2,
                   concepts1, concepts2,
                   title1=None, title2=None, use_cd=True,
                   betweenness_threshold_coef=1.0, max_c_size=10, min_c_size=3, IDF=None):
    """
    Given a pair of segmented texts and concepts lists,
    get ccig.
    :param text1: segmented text.
    :param concepts1: a list of concept words.
    :param text2: segmented text.
    :param concepts2: a list of concept words.
    :return: sentence lists of text1 and text2, and ccig.
    """
    sentences1 = split_sentence(text1, LANGUAGE)
    sentences2 = split_sentence(text2, LANGUAGE)
    g = construct_align_ccig(sentences1, sentences2,
                             concepts1, concepts2,
                             title1, title2, use_cd,
                             betweenness_threshold_coef, max_c_size, min_c_size, IDF)
    return g, sentences1, sentences2


def construct_cig(sentences, concepts):
    """
        Given a segmented text and a list of concepts,
        construct concept interaction graph.
        :param sentences: a list of sentences.
        :param concepts: a list of strings.
        :return: a concept interaction graph.
        """
    g = nx.Graph()
    concepts = list(set(concepts))
    if len(sentences) == 0:
        return g

    vprop_name = {}
    vprop_sentidxs = {}
    vertex_idx_map = {}

    v_empty = 0
    g.add_node(v_empty)
    vprop_name[v_empty] = EMPTY_VERTEX_NAME
    vprop_sentidxs[v_empty] = []
    vertex_idx_map[EMPTY_VERTEX_NAME] = 0

    g.node[v_empty]['name'] = vprop_name[v_empty]
    g.node[v_empty]['sentidxs'] = vprop_sentidxs[v_empty]

    i = 1
    for concept in concepts:
        g.add_node(i)
        vprop_name[i] = concept
        vprop_sentidxs[i] = []
        g.node[i]['name'] = vprop_name[i]
        g.node[i]['sentidxs'] = vprop_sentidxs[i]
        vertex_idx_map[concept] = i
        i = i + 1

    # create edges
    eprop_name = {}
    eprop_sentidxs = {}
    eprop_weight = {}

    for i in range(len(sentences)):
        sent = sentences[i]
        words = str(sent).split()
        is_sent_contain_concept = False
        for w in words:
            if w in concepts:
                is_sent_contain_concept = True
                vprop_sentidxs[vertex_idx_map[w]].append(i)
                g.node[vertex_idx_map[w]]['sentidxs'].append(i)
            if not is_sent_contain_concept:
                vprop_sentidxs[v_empty].append(i)
                g.node[v_empty]['sentidxs'].append(i)

    for i in range(len(concepts)):
        vi_idx = vertex_idx_map[concepts[i]]
        for j in range(i + 1, len(concepts)):
            vj_idx = vertex_idx_map[concepts[j]]
            common_sentidxs = common(vprop_sentidxs[vi_idx],
                                     vprop_sentidxs[vj_idx])
            # print(common_sentidxs)
            if len(common_sentidxs) > 0:
                e = (vi_idx, vj_idx)
                g.add_edge(vi_idx, vj_idx)

                eprop_sentidxs[e] = common_sentidxs
                eprop_weight[e] = len(common_sentidxs)

                if concepts[i] < concepts[j]:
                    eprop_name[e] = concepts[i] + '_' + concepts[j]
                else:
                    eprop_name[e] = concepts[j] + '_' + concepts[i]

                g[vi_idx][vj_idx]['sentidxs'] = eprop_sentidxs[e]
                g[vi_idx][vj_idx]['weight'] = eprop_weight[e]
                g[vi_idx][vj_idx]['name'] = eprop_name[e]

    for i in range(len(concepts)):
        vi_idx = vertex_idx_map[concepts[i]]

        for vj_idx in g.nodes():
            if vi_idx == vj_idx:
                continue
            if g.has_edge(vi_idx, vj_idx):
                # print(vprop_sentidxs[vi_idx])
                # print(eprop_sentidxs[(vi_idx, vj_idx)])
                e = (vi_idx, vj_idx)

                if e in eprop_sentidxs.keys():
                    #print(eprop_sentidxs[e])
                    # vprop_sentidxs[vi_idx] = substract(vprop_sentidxs[vi_idx], \
                    #                                    eprop_sentidxs[e])

                    g.node[vi_idx]['sentidxs'] = vprop_sentidxs[vi_idx]
    # for n, nbrs in g.adjacency():
    #     for nbr, eattr in nbrs.items():
    #         data1 = eattr
    #         # data2 = eattr['eprop_concepts']
    #         # data3 = eattr['eprop_weight_numsent']
    #         # data4 = eattr['eprop_weight_numsent']
    #         print(n, nbr, data1)
    #graph properties
    g.graph['numsent'] = float(len(sentences))

    return g

def get_concept_communities(sentences, concepts, betweenness_threshold_coef, max_c_size, min_c_size):
    """
    Given a segmented text and a list of concepts,
    construct concept graph and get concept communities.
    :param sentences: a list of sentences.
    :param concepts: a list of concepts.
    :return: a list of concept communities.
    """
    # cig = construct_ccig(sentences, concepts)
    cig = construct_cig(sentences, concepts)
    from networkx.algorithms import community
    communities_generator = community.girvan_newman((cig))
    top_level_communities = next(communities_generator)
    #next_level_communities = next(communities_generator)

    concept_communities = []
    for commuities in top_level_communities:

        vs_names = [cig.node[vi]['name'] for vi in commuities]
        concept_communities.append(vs_names)

    concept_communities.remove([EMPTY_VERTEX_NAME])

    return concept_communities





def community2name(community):
    """
    Given a list of concepts (words), return a string as its name.
    """
    return " ".join(community)


def name2community(name):
    """
    Given the string name of a community, recover the community.
    """
    return str(name).split()


def assign_sentences_to_concept_communities(sentences, concept_communities, IDF):
    """
    Assign a list of sentences to different concept communities.
    :param sentences: a list of sentences
    :param concept_communities: a list of concept lists.
    :return: a dictionary of (community_name, sentence index list)
    """
    names = [community2name(c) for c in concept_communities]
    name_sentidxs = {}
    name_sentidxs[EMPTY_VERTEX_NAME] = []
    for name in names:
        name_sentidxs[name] = []
    for i in range(len(sentences)):
        # NOTICE: make sure IDF contains all concepts
        scores = [tfidf_cos_sim(sentences[i], name, IDF) for name in names]
        max_index = scores.index(max(scores))
        max_score = max(scores)
        if max_score > 0:
            name_sentidxs[names[max_index]].append(i)
        else:
            name_sentidxs[EMPTY_VERTEX_NAME].append(i)
    return name_sentidxs


def assign_sentences_to_concepts(sentences, concepts):
    """
    Assign a list of sentences to different concept communities.
    :param sentences: a list of sentences
    :param concepts: a list of concepts.
    :return: a dictionary of (concept_name, sentence index list)
    """
    concept_sentidxs = {}
    concept_sentidxs[EMPTY_VERTEX_NAME] = []
    assigned_sentidxs = []

    for concept in concepts:
        concept_sentidxs[concept] = []
        for i in range(len(sentences)):
            words = str(sentences[i]).split()

            if concept in words:
                concept_sentidxs[concept].append(i)
                assigned_sentidxs.append(i)

    concept_sentidxs[EMPTY_VERTEX_NAME] = [x for x in range(len(sentences))
                                           if x not in assigned_sentidxs]
    return concept_sentidxs


if __name__ == "__main__":
    text = "中国 在 看 日本 。 \
         韩国 不听 中国 。 \
         中国 靠近 俄国 。 \
         俄国 远离 韩国 。 \
         韩国 仇视 日本 。 \
         日本 亲近 美国 。 \
         日本 亲近 美国 。 \
         日本 亲近 美国 。 \
         美国 和 加拿大 接壤 。 \
         加拿大 不熟悉 泰国 。 \
         美国 不鸟 泰国 。 \
         泰国 靠近 越南 。 \
         越南 靠近 老挝 。 \
         泰国 靠近 老挝 。 \
         美国 自己 玩 。 \
         新加坡 非常 小 。 \
         新加坡 靠近 中国 。\
         哈哈 哈哈"
    concepts = ["中国", "日本", "韩国", "美国", "新加坡",
                "俄国", "泰国", "老挝", "越南", "加拿大", ""]

    sentences = split_sentence(text, LANGUAGE)
    # print(sentences)
    # g = construct_ccig(sentences, concepts, use_cd=True)
    IDF = load_IDF("event_story")
    g = construct_ccig(sentences, concepts, use_cd=True, IDF=IDF)

    #g = construct_cig(sentences, concepts)

    #print(get_concept_communities(sentences, concepts, 1.0, 10, 6))
    #from networkx.algorithms import community
    # communities_generator = community.girvan_newman((g))
    # top_level_communities = next(communities_generator)
    # next_level_communities = next(communities_generator)
    # print(top_level_communities[0])
    # print(next_level_communities)
    # nx.draw(g)
    # plt.show()
    for n, nbrs in g.adjacency():
        for nbr, eattr in nbrs.items():
            data1 = eattr
            print(n, nbr, data1)

    for v in g.nodes():
        print(g.node[v])

    # print(assign_sentences_to_concepts(split_sentence(text, LANGUAGE), concepts))
