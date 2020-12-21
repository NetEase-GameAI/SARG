import os
import torch
from transformers import BertModel
from modeling_sarg import SARGModel, SARGConfig


def convert_weight_bert_to_ptrgen(vocab_size, n_utterance, load_bert_path, dump_sarg_path):
    bert_model = BertModel.from_pretrained(load_bert_path)
    sarg_model = SARGModel(SARGConfig(vocab_size=vocab_size, n_utterance=n_utterance))
    with torch.no_grad():
        bnp = {k: v for k, v in bert_model.named_parameters()}
        for name, param in sarg_model.named_parameters():
            if 'word_embeddings' in name:
                param.copy_(bnp['embeddings.word_embeddings.weight'])
            elif 'pos_embeddings' in name:
                param.copy_(bnp['embeddings.position_embeddings.weight'][:sarg_model.config.n_positions])
            elif 'embed_layer_norm.weight' in name:
                param.copy_(bnp['embeddings.LayerNorm.weight'])
            elif 'embed_layer_norm.bias' in name:
                param.copy_(bnp['embeddings.LayerNorm.bias'])
            elif 'attn.c_attn.weight' in name:
                layer_id = name.split('.')[1]
                qw = bnp['encoder.layer.%s.attention.self.query.weight' % layer_id]
                kw = bnp['encoder.layer.%s.attention.self.key.weight' % layer_id]
                vw = bnp['encoder.layer.%s.attention.self.value.weight' % layer_id]
                qkvw = torch.cat([qw, kw, vw], dim=0)  # (3 * dim, dim)
                param.copy_(qkvw)
            elif 'attn.c_attn.bias' in name:
                layer_id = name.split('.')[1]
                qb = bnp['encoder.layer.%s.attention.self.query.bias' % layer_id]
                kb = bnp['encoder.layer.%s.attention.self.query.bias' % layer_id]
                vb = bnp['encoder.layer.%s.attention.self.query.bias' % layer_id]
                qkvb = torch.cat([qb, kb, vb], dim=-1)
                param.copy_(qkvb)
            elif 'attn.c_proj.weight' in name:
                layer_id = name.split('.')[1]
                param.copy_(bnp['encoder.layer.%s.attention.output.dense.weight' % layer_id])
            elif 'attn.c_proj.bias' in name:
                layer_id = name.split('.')[1]
                param.copy_(bnp['encoder.layer.%s.attention.output.dense.bias' % layer_id])
            elif 'ln_1.weight' in name:
                layer_id = name.split('.')[1]
                param.copy_(bnp['encoder.layer.%s.attention.output.LayerNorm.weight' % layer_id])
            elif 'ln_1.bias' in name:
                layer_id = name.split('.')[1]
                param.copy_(bnp['encoder.layer.%s.attention.output.LayerNorm.bias' % layer_id])
            elif 'ff.zoomin.weight' in name:
                layer_id = name.split('.')[1]
                param.copy_(bnp['encoder.layer.%s.intermediate.dense.weight' % layer_id])
            elif 'ff.zoomin.bias' in name:
                layer_id = name.split('.')[1]
                param.copy_(bnp['encoder.layer.%s.intermediate.dense.bias' % layer_id])
            elif 'ff.zoomout.weight' in name:
                layer_id = name.split('.')[1]
                param.copy_(bnp['encoder.layer.%s.output.dense.weight' % layer_id])
            elif 'ff.zoomout.bias' in name:
                layer_id = name.split('.')[1]
                param.copy_(bnp['encoder.layer.%s.output.dense.bias' % layer_id])
            elif 'ln_2.weight' in name:
                layer_id = name.split('.')[1]
                param.copy_(bnp['encoder.layer.%s.output.LayerNorm.weight' % layer_id])
            elif 'ln_2.bias' in name:
                layer_id = name.split('.')[1]
                param.copy_(bnp['encoder.layer.%s.output.LayerNorm.bias' % layer_id])
        if not os.path.exists(dump_sarg_path):
            os.mkdir(dump_sarg_path)
        sarg_model.save_pretrained(dump_sarg_path)


if __name__ == "__main__":
    # for chinese dataset
    pretrained_model_path = "chinese_roberta_wwm_ext_pytorch"
    sarg_init_model_path = "Roberta-WWM-Init"
    convert_weight_bert_to_ptrgen(
            vocab_size=21128,
            n_utterance=10,
            load_bert_path=pretrained_model_path, 
            dump_sarg_path=sarg_init_model_path
    )
    
    # for english dataset
    pretrained_model_path = "bert-base-uncased"
    sarg_init_model_path = "bert-bsae-uncased-init"
    convert_weight_bert_to_ptrgen(
            vocab_size=30522,
            n_utterance=30,
            load_bert_path=pretrained_model_path, 
            dump_sarg_path=sarg_init_model_path
    )

