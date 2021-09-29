# A Knowledge Graph based Approach for Metaphor Processing
 This project include the dataset and source code for the paper Knowledge graph embedding approach for metaphor processing. 
 
## The main idea
![image](https://user-images.githubusercontent.com/42745094/135192649-9123abc2-7d83-4987-8fe3-39a5422104db.png)

Metaphor is a figure of speech that describes one thing (a target) by mentioning another thing (a source) in a way that is not literally true. Metaphor understanding is an interesting but challenging problem in natural language processing. This paper presents a novel method for metaphor processing based on knowledge graph (KG) embedding. Conceptually, we abstract the structure of a metaphor as an attribute-dependent relation between the target and the source. Each specific metaphor can be represented as a metaphor triple (target, attribute, source). Therefore, we can model metaphor triples just like modeling fact triples in a KG and exploit KG embedding techniques to learn better representations of concepts, attributes and concept relations. In this way, metaphor interpretation and generation could be seen as KG completion, while metaphor detection could be viewed as a representation learning enhanced concept pair classification problem. Technically, we build a Chinese metaphor KG in the form of metaphor triples based on simile recognition, and also extract concept-attribute collocations to help describe concepts and measure concept relations. We extend the translation-based and the rotation-based KG embedding models to jointly optimize metaphor KG embedding and concept-attribute collocation embedding. Experimental results demonstrate the effectiveness of our method. Simile recognition is feasible for building the metaphor triple resource. The proposed models improve the performance on metaphor interpretation and generation, and the learned representations also benefit nominal metaphor detection compared with strong baselines.



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
```bibtex
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
