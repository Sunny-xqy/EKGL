import xlrd
import pandas as pd
import numpy as np
import json



def load_training_data(f_name,node):
    print('We are loading data from:', f_name)
    # edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    with open(f_name, 'r', encoding='utf-8') as f:
        for line in f:
            words = line[:-1].split(' ')
            # if words[0] not in edge_data_by_type:
            #     edge_data_by_type[words[0]] = list()
            x, y = words[1], words[2]
            # edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
            if x == node or y ==node:
                all_edges.append(words)
    all_nodes = list(set(all_nodes))
    # all_edges = list(set(all_edges))
    # edge_data_by_type['Base'] = all_edges
    # print('Total training nodes: ' + str(len(all_nodes)))
    return all_nodes,all_edges


def data_new_end():
    f2 = open( 'data/ACT_r/name2node.json', 'r', encoding='utf-8')
    dictionary = json.load(f2)
    data_links = pd.read_csv('data/ACT_r/datasource/relations.csv')
    ftest = open( 'data/ACT_r/test.txt', 'w', encoding='utf-8')
    fvaild = open( 'data/ACT_r/valid.txt', 'w', encoding='utf-8')
    with open( 'data/ACT_r/train.txt', 'w', encoding='utf-8') as f:
        links={"1":[],"2":[],"3":[],"6":[]}
        label_count_test = {"1": 0, "2": 0, "3": 0, "6": 0}
        label_count_valid = {"1": 0, "2": 0, "3": 0, "6": 0}
        for i in range(len(data_links['stock'])):
            entity1 = trans(dictionary, data_links['entity1'][i])
            entity2 = trans(dictionary, data_links['entity2'][i])
            choose = np.random.choice(range(0,25))
            if choose == 1 :
                ftest.write(str(data_links['relations'][i]) + ' ' + str(entity1) + ' ' + str(entity2) + ' ' + '1' + '\n')
                label_count_test[str(data_links['relations'][i])] = label_count_test[str(data_links['relations'][i])] +1
            elif choose == 0 :
                fvaild.write(str(data_links['relations'][i]) + ' ' + str(entity1) + ' ' + str(entity2) + ' ' + '1' + '\n')
                label_count_valid[str(data_links['relations'][i])] = label_count_valid[str(data_links['relations'][i])] + 1
            else:
                print(choose)
                f.write(str(data_links['relations'][i]) + ' ' + str(entity1) + ' ' + str(entity2) + '\n')
                f.write(str(data_links['relations'][i]) + ' ' + str(entity2) + ' ' + str(entity1) + '\n')
            links[str(data_links['relations'][i])].append((str(entity1),str(entity2)))
        print(label_count_valid)
        print(list(dictionary.values()))
        for j in ['1','2','3','6']:
            for i in range(label_count_test[j]):
                entity1=np.random.choice(list(dictionary.values()))
                entity2=np.random.choice(list(dictionary.values()))
                if entity1 !=entity2 and (str(entity1),str(entity2)) not in links[j]:
                    ftest.write(j + ' ' + str(entity1) + ' ' + str(entity2) + ' ' + '0' + '\n')
        for j in ['1','2','3','6']:
            for i in range(label_count_valid[j]):
                entity1 = np.random.choice(list(dictionary.values()))
                entity2 = np.random.choice(list(dictionary.values()))
                if entity1 != entity2 and (str(entity1), str(entity2)) not in links[j]:
                    fvaild.write(j + ' ' + str(entity1) + ' ' + str(entity2) + ' ' + '0' + '\n')


def trans( dict, key):
    return dict[key]


if __name__ == '__main__':
    data_new_end()
