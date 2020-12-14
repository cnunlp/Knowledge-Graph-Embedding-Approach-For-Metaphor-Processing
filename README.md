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
@ARTICLE{9292078,  
    author={W. {Song} and 
    J. {Guo} and 
    R. {Fu} and 
    T. {Liu} and 
    L. {Liu}},  
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},   
    title={A Knowledge Graph Embedding Approach for Metaphor Processing},   
    year={2020},  
    volume={},  
    number={},  
    pages={1-1},  
    doi={10.1109/TASLP.2020.3040507}}
```