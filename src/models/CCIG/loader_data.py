# coding=utf-8
import numpy as np
import scipy.sparse as sp
import json
from sklearn.preprocessing import OneHotEncoder
from util.nlp_utils import *


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def load_graph_data(path, word_to_ix, max_len, num_data):
    """
    :param path: graph data file path
    :param word_to_ix: transform the word sentence the sequence
    :param max_len: maximum length of text extracted from one document in each vertice
    :return:
    """
    labels_list = []  # doc pair relationship labels

    g_features_list = []  # doc pair global feature vectors
    g_multi_scale_features_list = []  # doc pair multi-scale feature vectors
    g_all_features_list = []  # doc pair all feature vectors

    g_vertices_betweenness_list = []  # betweenness scores of vertices
    g_vertices_pagerank_list = []  # page rank scores of vertices
    g_vertices_katz_list = []  # katz scores of vertices

    v_features_list = []
    v_texts_w2v_idxs_l_list = []
    v_texts_w2v_idxs_r_list = []

    adjs_numsent_list = []
    adjs_tfidf_list = []
    adjs_position_list = []
    adjs_textrank_list = []

    num_samples = 0

    fin = open(path, "r", encoding="utf-8")
    for line in fin:  # each line is a json data that stores a CCIG/CIG graph
        g = json.loads(line)

        # labels
        labels_list.append(g["label"])

        # global doc pair features
        g_features_list.append(g["g_features_vec"])  # similarities
        g_multi_scale_features_list.append(g["g_multi_scale_features_vec"])  # multi-scale similarities
        g_all_features_list.append(g["g_all_features_vec"])  # multi-scale similarities + time gap + topic

        # vertices features
        v_features_list.append(g["v_features_mat"])

        # vertices scores
        g_vertices_betweenness_list.append(g["g_vertices_betweenness_vec"])
        g_vertices_pagerank_list.append(g["g_vertices_pagerank_vec"])
        g_vertices_katz_list.append(g["g_vertices_katz_vec"])

        # TODO: handle non-w2v case, i.e., the features are just vertices' numerical features
        v_texts = g["v_texts_mat"]
        word_idxs = []
        for i, row in enumerate(v_texts):
            word_idxs.append([]) #row = [text1, text2]
            for j, val in enumerate(row):
                val = val.encode("utf8").split()
                sent_idx = right_pad_zeros_1d([word_to_ix[w.decode("utf-8")] for w in val], max_len)
                word_idxs[i].append(sent_idx)
        #[[text1, text2],
        # [......,......],
        # [text1, text2]]
        #print(word_idxs[0])
        word_idxs_l = [s[0] for s in word_idxs]
        word_idxs_r = [s[1] for s in word_idxs]
        v_texts_w2v_idxs_l_list.append(word_idxs_l)#[[v1_text1],[v2_text1],...[]]
        v_texts_w2v_idxs_r_list.append(word_idxs_r)



        # adjacent matrices
        adj_numsent = g["adj_mat_numsent"]
        adj_numsent = sp.coo_matrix(adj_numsent, shape=(len(adj_numsent), len(adj_numsent)), dtype=np.float32)
        adj_numsent = normalize(adj_numsent)
        #adj_numsent = sparse_mx_to_torch_sparse_tensor(adj_numsent)
        adjs_numsent_list.append(adj_numsent)

        adj_tfidf = g["adj_mat_tfidf"]
        #adj_tfidf = sp.coo_matrix(adj_tfidf, shape=(len(adj_tfidf), len(adj_tfidf)), dtype=np.float32)
        #adj_tfidf = normalize(adj_tfidf)
        #adj_tfidf = sparse_mx_to_torch_sparse_tensor(adj_tfidf)
        adjs_tfidf_list.append(adj_tfidf)

        adj_position = g["adj_mat_position"]
        #adj_position = sp.coo_matrix(adj_position, shape=(len(adj_position), len(adj_position)), dtype=np.float32)
        #adj_position = normalize(adj_position)
        #adj_position = sparse_mx_to_torch_sparse_tensor(adj_position)
        adjs_position_list.append(adj_position)

        adj_textrank = g["adj_mat_textrank"]
        #adj_textrank = sp.coo_matrix(adj_textrank, shape=(len(adj_textrank), len(adj_textrank)), dtype=np.float32)
        #adj_textrank = normalize(adj_textrank)
        #adj_textrank = sparse_mx_to_torch_sparse_tensor(adj_textrank)
        adjs_textrank_list.append(adj_textrank)

        num_samples = num_samples + 1

        if num_samples >= num_data:
            break

    #our test label

    # labels
    labels = labels_list

    labels = [[l] for l in labels]

    # global doc-pair features
    g_features_list = np.array(g_features_list)
    g_features = [np.array(x) for x in g_features_list]

    g_multi_scale_features_list = np.array(g_multi_scale_features_list)
    g_multi_scale_features = [np.array(x) for x in g_multi_scale_features_list]

    g_all_features_list = np.array(g_all_features_list)
    g_topic_feature = g_all_features_list[:, -1:]  # NOTICE: see feature_extractor.py, the last one is topic feature
    enc = OneHotEncoder()
    g_topic_feature = enc.fit_transform(g_topic_feature).toarray()
    g_all_features_list = np.concatenate((g_all_features_list[:, :-1], g_topic_feature), axis=1).tolist()
    g_all_features = [np.array(x) for x in g_all_features_list]

    # vertices scores
    g_vertices_betweenness = [np.array(x[:-1]) for x in g_vertices_betweenness_list]
    g_vertices_pagerank = [np.array(x[:-1]) for x in g_vertices_pagerank_list]
    g_vertices_katz = [np.array(x[:-1]) for x in g_vertices_katz_list]

    # split train:valid:test = 0.6:0.2:0.2
    train_bound = int(num_samples * 0.6)
    eval_bound = int(num_samples * (0.6 + 0.2))
    idx_train = range(train_bound)
    idx_val = range(train_bound, eval_bound)
    idx_test = range(eval_bound, num_samples)
    #idx_test = range(num_samples-107, num_samples)

    # print(v_texts_w2v_idxs_l_list[0])
    # print(v_texts_w2v_idxs_r_list[0])

    # print(len(v_texts_w2v_idxs_r_list))
    # print(len(v_texts_w2v_idxs_l_list))
    return v_texts_w2v_idxs_l_list, v_texts_w2v_idxs_r_list, v_features_list,\
        adjs_numsent_list, adjs_tfidf_list, adjs_position_list, adjs_textrank_list,\
        g_features, g_multi_scale_features, g_all_features,\
        g_vertices_betweenness, g_vertices_pagerank, g_vertices_katz,\
        labels, idx_train, idx_val, idx_test, train_bound, eval_bound,num_samples
