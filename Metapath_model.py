import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time
from utils import *
import tensorflow as tf
import json
import ast
from utils import *
from walk import RandomWalker
import argparse
import csv


class Metapath:
    def __init__(self, num_nodes,n_sampled ,num_feat,nodes, embedding_dim=200, lr=0.001):
        self.num_feat = num_feat
        self.n_sampled = n_sampled
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes
        self.lr = lr
        self.nodes = nodes 
        self.softmax_w = tf.Variable(tf.truncated_normal((num_nodes, embedding_dim), stddev=0.1), name='softmax_w')
        self.softmax_b = tf.Variable(tf.zeros(num_nodes), name='softmax_b')
        self.inputs = self.input_init()
        self.side_info = self.act_info()
        self.embedding = self.embedding_init()
        self.alpha_embedding = tf.Variable(tf.random_uniform((num_nodes, num_feat), -1, 1),name='alpha')
        self.merge_emb = self.attention_merge()
        self.cost = self.make_skipgram_loss()
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.cost)
        

    def act_info(self):
        start = time.time()
        side_embedding = []
        for i in range(self.num_feat-1):
            embedding_child = np.zeros((self.num_nodes,self.embedding_dim),dtype=float)
            side_embedding.append(embedding_child)
        print('Processing relations among shareholders......')
        att = np.load('embeddings/relation_embedds.npy', allow_pickle=True)
        relation_dict = att.item()
        f2 = open('data/ACT_r/node2name.json', 'r', encoding='utf-8')
        dictionary = json.load(f2)
        index_label=0
        for label in ['1','2','3','4','5','6']:
            for id in dictionary.keys():
                id_code = dictionary[id]
                try:
                    id_code_new = find_keys(self.nodes,id)
                    embedding=relation_dict[label][str(id_code)]
                    side_embedding[index_label][id_code_new[0],::] = embedding
                except:
                    pass
            index_label = index_label+1
        end = time.time()
        print("Time cost:time".format(time=end - start))
        return side_embedding

    def embedding_ini(self):
        """
        :return: Initial embedding
        """
        cat_embedding_vars = []
        for i in range(self.num_feat):
            embedding_var = tf.Variable(tf.random_uniform((self.feature_lens[i], self.embedding_dim), -1, 1), name='embedding'+str(i),
                                        trainable=True)
            cat_embedding_vars.append(embedding_var)
        return cat_embedding_vars

    def attention_merge(self):
        embed_list = []
        for i in range(self.num_feat):
            cat_embed = tf.nn.embedding_lookup(self.embedding[i], self.inputs[0])
            embed_list.append(cat_embed)
        stack_embed = tf.stack(embed_list, axis=-1)
        # attention merge
        alpha_embed = tf.nn.embedding_lookup(self.alpha_embedding, self.inputs[0])
        alpha_embed_expand = tf.expand_dims(alpha_embed, 1)
        alpha_i_sum = tf.reduce_sum(tf.exp(alpha_embed_expand), axis=-1)
        merge_emb = tf.reduce_sum(stack_embed * tf.exp(alpha_embed_expand), axis=-1) / alpha_i_sum
        return merge_emb

    def input_init(self):
        input_list = []
        for i in range(self.num_feat):
            input_col = tf.placeholder(tf.int32, [None], name='inputs_'+str(i))
            input_list.append(input_col)
        input_list.append(tf.placeholder(tf.int32, shape=[None, 1], name='label'))
        return input_list

    def make_skipgram_loss(self):
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
            weights=self.softmax_w,
            biases=self.softmax_b,
            labels=self.inputs[-1],
            inputs=self.merge_emb,
            num_sampled=self.n_sampled,
            num_classes=self.num_nodes,
            num_true=1,
            sampled_values=tf.random.uniform_candidate_sampler(
                true_classes=tf.cast(self.inputs[-1], tf.int64),
                num_true=1,
                num_sampled=self.n_sampled,
                unique=True,
                range_max=self.num_nodes
            )
        ))
        return loss
    
def get_graph_context_all_pairs(walks, window_size):
    all_pairs = []
    for k in range(len(walks)):
        for i in range(len(walks[k])):
            for j in range(i - window_size, i + window_size + 1):
                if i == j or j < 0 or j >= len(walks[k]):
                    continue
                else:
                    all_pairs.append([walks[k][i], walks[k][j]])
    return np.array(all_pairs)

def get_walks_mywalk(G):
    walker = RandomWalker(G, p=args.p, q=args.q)
    print("Preprocess transition probs...")
    walker.preprocess_transition_probs()

    session_reproduce = walker.simulate_walks(num_walks=args.num_walks, walk_length=args.walk_length, workers=4,
                                              verbose=1)
    session_reproduce = list(filter(lambda x: len(x) > 2, session_reproduce))

    all_pairs = get_graph_context_all_pairs(session_reproduce, args.window_size)
    
    return all_pairs 

def model_train(length,all_pairs,enc):
    EKGL = Metapath(length,args.n_sampled, args.num_feat,enc,embedding_dim=args.embedding_dim,
                      lr=args.lr)
    # init model
    print('init...')
    start_time = time.time()
    init = tf.global_variables_initializer()
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    sess = tf.Session(config=config_tf)
    sess.run(init)
    end_time = time.time()
    print('time consumed for init: %.2f' % (end_time - start_time))

    print_every_k_iterations = 100
    loss = 0
    iteration = 0
    start = time.time()

    max_iter = len(all_pairs) // args.batch_size * args.epochs
    # f  = open('EKGL_train.txt', 'w', encoding='utf-8')
    f  = open('system_train.txt', 'w', encoding='utf-8')
    for iter in range(max_iter):
        iteration += 1
        batch_features, batch_labels = next(graph_context_batch_iter(all_pairs, args.batch_size))
        feed_dict = {input_col: batch_features[:, i] for i, input_col in enumerate(EKGL.inputs[:-1])}
        feed_dict[EKGL.inputs[-1]] = batch_labels
        # _, train_loss = sess.run([EKGL.train_op, EKGL.cost], feed_dict=feed_dict)
        _, train_loss = sess.run([EKGL.train_op, EKGL.cost], feed_dict=feed_dict)

        loss += train_loss

        if iteration % print_every_k_iterations == 0:
            end = time.time()
            e = iteration * args.batch_size // len(all_pairs)
            print("Epoch {}/{}".format(e+1, args.epochs),
                  "Iteration: {}".format(iteration),
                  "Avg.Training_loss: {:.4f}".format(loss / print_every_k_iterations),
                  "Time: {:.4f} sec/batch".format((end - start) / print_every_k_iterations))
            f.write(str([iteration,loss / print_every_k_iterations]))
            loss = 0
            start = time.time()


    print('optimization finished...')
    saver = tf.train.Saver()
    # saver.save(sess, "checkpoints/EKGL")
    saver.save(sess, "checkpoints/EKGL")
    print('saving embedding result...')
    features=np.zeros((length, 1), dtype=np.int32)
    features[:,0] = np.arange(length)
    feed_dict = {input_col: features[:, i] for i, input_col in enumerate(EKGL.inputs[:-1])}
    feed_dict[EKGL.inputs[-1]] = np.zeros((length, 1), dtype=np.int32)
    embedding_result = sess.run(EKGL.merge_emb, feed_dict=feed_dict)
    write_embedding(embedding_result, args.outputEmbedFile)
    return embedding_result


def get_graph():
    f = open('/home/xuqingying/my_work/EKGL/data/Shareholder_r/shareholding.txt','r')
    data=[]
    for line in f:
        line = line[:-1]    
        links = ast.literal_eval(line)['links']
        if len(links)>3:
            data = data + links
    f.close()

    print('Start giving id for nodes, links......')
    nodes = {}
    links = []
    edges = []
    nodeid = 0

    link_path = "/home/xuqingying/my_work/EKGL/data/Shareholder_r/shareholdinglink.csv"
    edge_path = "/home/xuqingying/my_work/EKGL/data/Shareholder_r/shareholdingedge.csv"
    node_path = "/home/xuqingying/my_work/EKGL/data/Shareholder_r/shareholdingnode.json"
    for i in data:
        if i['source'] not in nodes:
            nodes[i['source']] = nodeid
            nodeid = nodeid + 1
        if i['target'] not in nodes:
            nodes[i['target']] = nodeid
            nodeid = nodeid + 1
        links.append([nodes[i['source']],nodes[i['target']]])
        edges.append([nodes[i['source']],nodes[i['target']],i['value']])
    print('Finished......')
    with open(node_path, 'w') as json_file:
        json.dump(nodes, json_file)
    with open(edge_path, 'w') as f:
        writer = csv.writer(f)
        for item in edges:
            writer.writerow(item)
    with open(link_path, 'w') as f:
        writer = csv.writer(f)
        for item in edges:
            writer.writerow(item)
    return nodes,edges,nodeid

def find_keys(dictionary, value):
    return [k for k, v in dictionary.items() if v == value]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--data_path", type=str, default='./data/')
    parser.add_argument("--p", type=float, default=1)
    parser.add_argument("--q", type=float, default=1)
    parser.add_argument("--num_walks", type=int, default=10)
    parser.add_argument("--walk_length", type=int, default=20)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--root_path", type=str, default='./data_cache/')
    parser.add_argument("--n_sampled", type=int, default=10)
    parser.add_argument("--num_feat", type=int, default=5)#属性维度+1
    parser.add_argument("--embedding_dim", type=int, default=200)
    parser.add_argument("--outputEmbedFile", type=str, default='/embedding/EKGL.embed')
    args = parser.parse_args()

    nodes,links,len_node=get_graph()
    len_node = len_node +1
    G = nx.DiGraph()
    G.add_weighted_edges_from(links)
    all_pairs=get_walks_mywalk(G)
    print(all_pairs)

    embedding_result = model_train(len_node,all_pairs,nodes)