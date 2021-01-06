# MetaphorProcessing
 This project include the dataset and source code for the paper Knowledge graph embedding approach for metaphor processing. Detail informatin please refers to the paper.

## Dataset
```
./data/metaphor_inter_and_gen    # metaphor interpretation and generation datasest 
./data/metahor_detection    # metaphor detection dataset
./data/TransHMetaJoint_embeddings    # concept_attribute embeddings
```
## Train and Test
```
./MetaphorInterAndGen/run.py    # training and testing code for metaphor interpretation and generation
./MetaphorIndentification/classifier.py    # training and testing code for metaphor indentification
./SimileRecognition/bert_simile_labeling.py    # training and testing code for simile recognition
./SimileRecognition/bert_simile_classification.py    # training and testing code for simile classification
```
## Reference
The code and dataset are released with this paper:
```
@article{song2020knowledge,
  title={A Knowledge Graph Embedding Approach for Metaphor Processing},
  author={Song, Wei and Guo, Jingjin and Fu, Ruiji and Liu, Ting and Liu, Lizhen},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={29},
  pages={406--420},
  year={2020},
  publisher={IEEE}
}
```
