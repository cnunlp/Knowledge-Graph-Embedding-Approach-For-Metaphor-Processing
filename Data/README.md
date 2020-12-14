## Dataset 
We provide the datasets and resources used in our research.
There are three directories containing a total of ten files.

## metaphor_interpretation_and_generation
+ metaphor_knowledge_graph_initial_train.txt
	This file contains the metaphor triples in train set that are extracted manually from simile sentences in the form of (target, source, attribute).
	All the targets, sources and attributes are labeled with the word class tags that are defined in the Chinese synonymy thesaurus.
	The tags are used in synonymy based evaluation.
	
+ metaphor_knowledge_graph_initial_validation.txt
	This file contains metaphor triples in validation set that are extracted manually from simile sentences in the form of (target, source, attribute).
	All the targets, sources and attributes are labeled with the word class tags that are defined in the Chinese synonymy thesaurus.
	The tags are used in synonymy based evaluation.
	
+ metaphor_knowledge_graph_initial_test.txt
	This file contains metaphor triples in test set that are extracted manually from simile sentences in the form of (target, source, attribute).
	All the targets, sources and attributes are labeled with the word class tags that are defined in the Chinese synonymy thesaurus.
	The tags are used in synonymy based evaluation.

+ metaphor_knowledge_graph_expanded_train.txt
	This file contains metaphor triples that are expanded by the simile recognition model.
	The triples are also in the form of (target, source, attribute), and all the targets, sources and attributes are labeled with the word class tags that are defined in the Chinese synonymy thesaurus.

+ concept_attribute_collocations.txt
	This file contains concept-attribute collocations in form of (concept, attribute, count). All the concepts and attributes are labeled with the word class tags that are defined in the Chinese synonymy thesaurus.
	
+ simile_recognition_data.txt
	This file includes the simile and non-simile sentences. 
	Different sentences are segmented by a blank line.
	The first line for each sentence indicates the number of words in the sentence, followed by a character each line with a simile component label.
	We use the IOBES(Inside, Outside, Beginning, Ending, Single) schema and use different prefixes to distinguish different components, i.e, 
	tb: the first character of a target;
	ti: the inner character of a target;
	te: the last character of a target;
	ts: a single character target;

	sb: the first character of a source;
	si: the inner character of a source;
	se: the last character of a source;
	ss: a single character source;

	apb: the first character of an adjective attribute;
	api: the inner character of an adjective attribute;
	ape: the last character of an adjective attribute;
	aps: a single character adjective attribute;

	vpb: the first character of a verb attribute;
	vpi: the inner character of a verb attribute;
	vpe: the last character of a verb attribute;
	vps: a single character verb attribute.

	O: others.

## metaphor_detection
+ metaphor_detection_train.txt
	This file contains concept pairs in train set with label 1 or 0, which means metaphorical pairs or non-metaphorical pairs respectively.
+ metaphor_detection_validation.txt
	This file contains concept pairs in validation set with label 1 or 0, which means metaphorical pairs or non-metaphorical pairs respectively.
	(3). metaphor_detection_test.txt
	This file contains concept pairs in test set with label 1 or 0, which means metaphorical pairs or non-metaphorical pairs respectively.
	
## TransMetaHjoint_embeddings
+ concept_attribute_embeddings.txt
	This file contains the embeddings of all concepts and attributes learned by TransMetaHjoint Model, the concept or attribute and its representation are segmented by the '\t'.


