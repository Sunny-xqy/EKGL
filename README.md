(1) We first run Multirelational/act_calculation.py to predict acting in concert relations among shareholders by the multirelations on EKG.
The relations are processed by data_process.py to obtain train.txt/valid..txt/test.txt;
Then, the follow parameter can be chose:
    | features | Input node attributes |
    | epoch | Traning epochs |
    | batch-size | Traning batch size |
    | schema | Walking schema |
    | dimensions | Number of dimensions of final embeddings |
    | edge-dim | Number of edge embedding dimensions |
    
To obtain the acting in concert relations, you can use python Multirelational/act_calculation.py -data/ACT_r with other chosing paramethers.

(2) We obtain the embeddings of entities by capture shareholding paths with considering acting in concert probability. To obtain embeddings, you can use python Metapath/Metapath_model.py -data/ACT_r with other chosing paramethers.
You can choose the follow parameter:
    | data_path | Data source |
    | num_walks | The number of walks for a node |
    | walk_length | The length of walk |
    | window_size | Training window of skip-gram method |

(3) We further train CoNN model by the ground truth of actual controller.
You can use CoNN_model.py to train it, and test it by test.py