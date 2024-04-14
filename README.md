(1) We first run Multirelational/act_calculation.py to predict acting in concert relations among shareholders by the multirelations on EKG.
The relations are processed by data_process.py to obtain train.txt/valid..txt/test.txt;
Then, the follow parameter can be chose:

    | Parameters | Description |
    | features | Input node attributes |
    | epoch | Traning epochs |
    | batch-size | Traning batch size |
    | schema | Walking schema |
    | dimensions | Number of dimensions of final embeddings |
    | edge-dim | Number of edge embedding dimensions |
    
To obtain the acting in concert relations, you can use:

```
python act_calculation.py -data/ACT_r
```

The predicted acting in concert probability will be perserved in "data/ACT_r/result.csv" and "data/ACT_r/other.csv"

(2) We obtain the embeddings of entities by capture shareholding paths with considering acting in concert probability. To obtain embeddings, you can use python Metapath/Metapath_model.py -data/ACT_r with other chosing paramethers.
You can choose the follow parameter:

    | Parameters | Description |
    | data_path | Data source |
    | num_walks | The number of walks for a node |
    | walk_length | The length of walk |
    | window_size | Training window of skip-gram method |
    
To obtain the generated embeddings of entities, you can use:

```
python Metapath_model.py
```
The generated embeddings be perserved in "embeddings/EKGL.embed"

(3) We further train CoNN model by the ground truth of actual controller.
You can use CoNN_model.py to train it, and test it by test.py
