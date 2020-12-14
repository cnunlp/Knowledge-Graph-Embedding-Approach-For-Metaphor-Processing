# Quick Start
```
python classifier.py \
--data_dir mc_data \
--output_dir res \
--pre_ent_embeds_path model.pt \
--pre_rel_embeds_path model.pt \
--pre_ten_embeds_path tencent_embedding/tencent_embeddings.txt \
--epoch 150 \
--batch_size 512 \
--embed_size 200 \
--hidden_size 256 \
--target_size 2 \
--optim_method Adam \
--lr 0.01 \
--embed_flag concat
```