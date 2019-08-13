# coding=utf-8
import time
import argparse
import math
from models.gcn_model import SE_GCN
from sklearn import metrics
from loader_data import *
# from models import *
# from models.ema import EMA
from collections import OrderedDict
from util.nlp_utils import *
import tensorflow as tf
import os
import datetime
import time
import logging
import numpy as np
from sklearn.metrics import matthews_corrcoef


# Training settings
parser = argparse.ArgumentParser()

# cuda, seed
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

# data files
parser.add_argument('--inputdata', type=str, default="event-story-cluster/same_story_doc_pair.cd.json",
                    help='input data path')
parser.add_argument('--outputresult', type=str, default="event-story-cluster/same_story_doc_pair.cd.result.txt",
                    help='output file path')
parser.add_argument('--data_type', type=str, default="event", help='event or story')

# train
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--num_data', type=int, default=1000000000, help='maximum number of data samples to use.')
parser.add_argument('--lr', default=0.0003, type=float, help='learning rate')
parser.add_argument('--lr_warm_up_num', default=1000, type=int, help='number of warm-up steps of learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='beta 1')
parser.add_argument('--beta2', default=0.999, type=float, help='beta 2')
parser.add_argument('--no_grad_clip', default=False, action='store_true', help='whether use gradient clip')
parser.add_argument('--max_grad_norm', default=5.0, type=float, help='global Norm gradient clipping rate')
parser.add_argument('--use_ema', default=False, action='store_true', help='whether use exponential moving average')
parser.add_argument('--ema_decay', default=0.9999, type=float, help='exponential moving average decay')

# model
parser.add_argument('--hidden_vfeat', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout_vfeat', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden_siamese', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--dropout_siamese', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden_final', type=int, default=16, help='Number of hidden units.')

parser.add_argument('--use_gcn', action='store_true', default=False, help='use GCN in model.')
parser.add_argument('--num_gcn_layers', type=int, default=2, help='number of gcn layers')
parser.add_argument('--use_cd', action='store_true', default=False, help='use community detection in graph.')
parser.add_argument('--use_siamese', action='store_true', default=False, help='use siamese encoding for vertices.')
parser.add_argument('--use_vfeatures', action='store_true', default=False, help='use vertex features.')
parser.add_argument('--use_gfeatures', action='store_true', default=False, help='use global graph features.')

parser.add_argument('--gfeatures_type', type=str, default="features",
                    help='what features to use: features, multi_scale_features, all_features')
parser.add_argument('--gcn_type', type=str, default="valina", help='gcn layer type')
parser.add_argument('--pool_type', type=str, default="mean", help='pooling layer type')
parser.add_argument('--combine_type', type=str, default="separate",
                    help='separate or concatenate, vfeatures with siamese encoding')

# graph
parser.add_argument('--adjacent', type=str, default="tfidf", help='adjacent matrix')
parser.add_argument('--vertice', type=str, default="pagerank", help='vertex centrality')
parser.add_argument('--betweenness_threshold_coef', type=float, default=1.0, help='community detection parameter')
parser.add_argument('--max_c_size', type=int, default=6, help='community detection parameter')
parser.add_argument('--min_c_size', type=int, default=2, help='community detection parameter')

args = parser.parse_args()

# data
print("begin loading W2V..........")
LANGUAGE = "Chinese"
W2V = load_w2v("../../../data/raw/word2vec/w2v-zh.model", "Company", 200)
W2V = dict((k, W2V[k]) for k in W2V.keys() if len(W2V[k]) == 200)
W2V = OrderedDict(W2V)


word_to_ix = {word: i for i, word in enumerate(W2V)}
MAX_LEN = 200  # maximum text length for each vertex
embed_size = len(list(W2V.values())[0])
print("W2V loaded! \nVocab size: %d, Embedding size: %d" % (len(word_to_ix), embed_size))

W2V = np.array(list(W2V.values()))

if args.data_type == "event" and args.use_cd:
    args.inputdata = "event-story-cluster/same_event_doc_pair.cd.json"
    args.outputresult = "event-story-cluster/same_event_doc_pair.cd.result.txt"
if args.data_type == "event" and not args.use_cd:
    args.inputdata = "event-story-cluster/same_event_doc_pair.no_cd.json"
    args.outputresult = "event-story-cluster/same_event_doc_pair.no_cd.result.txt"
if args.data_type == "story" and args.use_cd:
    args.inputdata = "event-story-cluster/same_story_doc_pair.cd.json"
    args.outputresult = "event-story-cluster/same_story_doc_pair.cd.result.txt"
if args.data_type == "story" and not args.use_cd:
    args.inputdata = "event-story-cluster/same_story_doc_pair.no_cd.json"
    args.outputresult = "event-story-cluster/same_story_doc_pair.no_cd.result.txt"
print(args)

path = "../../../data/processed/" + args.inputdata
print("begin loading DATA............" + path)

v_texts_w2v_idxs_l_list, v_texts_w2v_idxs_r_list, v_features_list, \
adjs_numsent_list, adjs_tfidf_list, adjs_position_list, adjs_textrank_list, \
g_features, g_multi_scale_features, g_all_features, \
g_vertices_betweenness, g_vertices_pagerank, g_vertices_katz, \
labels, idx_train, idx_val, idx_test, train_bound, eval_bound, num_samples = load_graph_data(path, word_to_ix, MAX_LEN, args.num_data)

print("DATA loaded! \nnumber of samples: %d, max sentence length: %d" % (
len(idx_train) + len(idx_val) + len(idx_test), MAX_LEN))

if args.gfeatures_type == "multi_scale_features":
    g_features = g_multi_scale_features
if args.gfeatures_type == "all_features":
    g_features = g_all_features

adjacent_dict = {
    "numsent": adjs_numsent_list,
    "tfidf": adjs_tfidf_list,
    "position": adjs_position_list,
    "textrank": adjs_textrank_list}
g_vertices_dict = {
    "betweenness": g_vertices_betweenness,
    "pagerank": g_vertices_pagerank,
    "katz": g_vertices_katz}
adjs = adjacent_dict[args.adjacent]
g_vertices = g_vertices_dict[args.vertice]

# model

dataset_id = "segcn-cache"
print('dataset_id:', dataset_id)

#logging
for handle in logging.root.handlers[:]:
    logging.root.removeHandler(handle)

logging.basicConfig(filename=dataset_id, format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.DEBUG, filemode='a', datefmt='%Y-%m-%d%I:%M:%S %p')

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement = True,
        log_device_placement = False
    )

    sess = tf.Session(config=session_conf)

    print("len(v_features_list[0]): ",len(v_features_list[0][0]))
    with sess.as_default():
        model = SE_GCN(
            args, W2V, sent_max_len=MAX_LEN, vector_size=embed_size, nfeat_v=len(v_features_list[0][0]),
            nfeat_g=len(g_features[0]),
            nhid_vfeat=args.hidden_vfeat, nhid_siamese=args.hidden_siamese, dropout_vfeat=args.dropout_vfeat,
            dropout_siamese=args.dropout_siamese,
            nhid_final=args.hidden_final,
            v_feature_shape=len(v_features_list[0][0]), g_feature_shape=len(g_features[0]),
        )

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta1=args.beta1, beta2=args.beta2,
                                           epsilon=1e-8)

        grads_and_vars = optimizer.compute_gradients(model.loss)
        capped_gvs = [(tf.clip_by_value(grad, -args.max_grad_norm, args.max_grad_norm), var) for grad, var in grads_and_vars]
        train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)


        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name),
                                                     tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        out_dir = os.path.abspath(os.path.join(os.path.curdir,
                                               'runs', dataset_id,
                                               'se_gcn'))

        print("Writing to {}\n".format(out_dir))

        if tf.gfile.Exists(out_dir):
            print('cleaning ', out_dir)
            tf.gfile.DeleteRecursively(out_dir)
        tf.gfile.MakeDirs(out_dir)

        loss_summary = tf.summary.scalar("loss", model.loss)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

def train_step(epoch, step):

    t = time.time()
    loss_train = 0.0
    scores_all=[]

    for i in idx_train:

        #print(v_texts_w2v_idxs_l_list[i])
        #print(v_features_list[i])
        #print(len(g_features[i]))
        #print(adjs[i])
        #print(labels[i])
        feed_dict = {
            model.input_l: v_texts_w2v_idxs_l_list[i],
            model.input_r: v_texts_w2v_idxs_r_list[i],
            model.v_features: v_features_list[i],
            model.g_feature: g_features[i],
            model.adj: adjs[i],
            model.input_y: labels[i],
        }
        _, step, summaries, loss, scores = sess.run(
            [train_op, global_step, train_summary_op, model.loss, model.score],
            feed_dict=feed_dict)
        train_summary_writer.add_summary(summaries, step)

        loss_train += loss
        scores_all.append(scores)

        step +=1
    loss_train = loss_train/len(idx_train)
    y_pred = np.array([1 if s >= 0.5 else 0 for s in scores_all])
    y_true = [labels[i][0] for i in idx_train]
    #print(y_true)
    acc_train = metrics.accuracy_score(y_true, y_pred)
    f1_train = metrics.f1_score(y_true, y_pred, average='micro')


    loss_val, acc_val, f1_val, outputs_val = dev()

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train),
          'acc_train: {:.4f}'.format(acc_train),
          'f1_train: {:.4f}'.format(f1_train),
          'loss_val: {:.4f}'.format(loss_val),
          'acc_val: {:.4f}'.format(acc_val),
          'f1_val: {:.4f}'.format(f1_val),
          'time: {:.4f}s'.format(time.time() - t))
    fout.write('Epoch: ' + str(epoch + 1) + ',' +
               'loss_train: ' + str(loss_train) + ',' +
               'acc_train: ' + str(acc_train) + ',' +
               'f1_train: ' + str(f1_train) + ',' +
               'loss_val: ' + str(loss_val) + ',' +
               "acc_val: " + str(acc_val) + ',' +
               'f1_val: ' + str(f1_val) + '\n')

    return step

def train(args, fout):

    step = 0
    for epoch in range(args.epochs):

        print("Epoch %d start..." % epoch)
        logging.info('Epoch %d start...' % epoch)
        start = time.time()

        step = train_step(epoch, step)

        end = time.time()
        print('Epoch %d Finshed: time %s' % (epoch, end - start))
        logging.info('Epoch %d Finshed: time %s' % (epoch, end - start))

        test_loss, test_acc, test_f1, outputs_test = test()
        print(
            "Test set results:",
            "loss= {:.4f}".format(test_loss),
            "accuracy= {:.4f}".format(test_acc),
            "f1= {:.4f}".format(test_f1))
        fout.write(
            "Test set results:" +
            "loss= " + str(test_loss) + ',' +
            "accuracy= " + str(test_acc) + ',' +
            "f1_score= " + str(test_f1) + '\n')

def dev(writer=None):
    scores_all = []
    losses_all = []

    for i in idx_val:
        feed_dict = {
            model.input_l: v_texts_w2v_idxs_l_list[i],
            model.input_r: v_texts_w2v_idxs_r_list[i],
            model.v_features: v_features_list[i],
            model.g_feature: g_features[i],
            model.adj: adjs[i],
            model.input_y: labels[i],
        }
        step, summaries, loss, scores = sess.run(
            [global_step, dev_summary_op, model.loss, model.score],
            feed_dict)

        scores_all.append(scores)
        losses_all.append(loss)

    #scores = np.concatenate(scores_all, axis=0)
    loss = np.mean(losses_all)

    time_str = datetime.datetime.now().isoformat()
    # print("[DEV] {}: step {}, loss {:g}".format(
    #     time_str, step, loss))
    logging.info("[DEV] {}: step {}, loss {:g}".format(
        time_str, step, loss))

    if writer:
        writer.add_summary(summaries, step)

    y_pred = np.array([1 if s >= 0.5 else 0 for s in scores_all])

    y_true = [labels[i][0] for i in idx_val]
    #print(y_true)
    acc_val = metrics.accuracy_score(y_true, y_pred)
    f1_val = metrics.f1_score(y_true, y_pred, average='micro')

    return loss, acc_val, f1_val, scores_all


def test(writer=None):
    scores_all = []
    losses_all = []
    for i in idx_test:
        feed_dict = {
            model.input_l: v_texts_w2v_idxs_l_list[i],
            model.input_r: v_texts_w2v_idxs_r_list[i],
            model.v_features: v_features_list[i],
            model.g_feature: g_features[i],
            model.adj: adjs[i],
            model.input_y: labels[i],
        }
        step, summaries, loss, scores = sess.run(
            [global_step, dev_summary_op, model.loss, model.score],
            feed_dict)

        scores_all.append(scores)
        losses_all.append(loss)

    # scores = np.concatenate(scores_all, axis=0)
    loss = np.mean(losses_all)

    time_str = datetime.datetime.now().isoformat()
    print("[DEV] {}: step {}, loss {:g}".format(
        time_str, step, loss))
    logging.info("[DEV] {}: step {}, loss {:g}".format(
        time_str, step, loss))

    if writer:
        writer.add_summary(summaries, step)

    y_pred = np.array([1 if s >= 0.5 else 0 for s in scores_all])
    y_true = [labels[i][0] for i in idx_test]
    acc_val = metrics.accuracy_score(y_true, y_pred)
    f1_val = metrics.f1_score(y_true, y_pred, average='micro')

    return loss, acc_val, f1_val, scores_all

def write_to_file(fin, label_list, pred_list):
    with open(fin, 'w') as f:
        for i in range(len(label_list)):
            f.write(str(label_list[i]) + '\t' + str(pred_list[i]) + '\n')

if __name__=='__main__':
    t_total = time.time()
    outpath = "../../../data/result/" + args.outputresult
    fout = open(outpath, 'w')
    train(args, fout)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    fout.close()