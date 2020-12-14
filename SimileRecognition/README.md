# Quick Start
## Simile Sentences Classification
```
python bert_simile_classification \
--data_dir data \
--bert_model model \
--task_name simile \
--output_dir res \
--cache_dir cache \
--max_seq_length 100 \
--train_batch_size 8 \
-eval_batch_size 3 \
--num_train_epochs 5 \
--do_classify
```

## Simile Component Extraction
```
python bert_simile_labeling.py \
--data_dir data \
--bert_model bert-base-chinese \
--task_name simile \
--output_dir res \
--cache_dir cache \
--max_seq_length 505 \
--beta 0.7 \
--gamma 0.3 \
--train_batch_size 8 \
--eval_batch_size 8 \
--num_train_epochs 20 \
--do_train \
--do_eval
```