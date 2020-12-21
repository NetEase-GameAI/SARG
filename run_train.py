import torch
import logging
import os
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional
from torch.utils.data import Dataset
import pickle
from overrides import overrides
from finetune_trainer import Trainer
from transformers import (
    BertTokenizer,
    DataCollator,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from modeling_sarg import (
    SARGModel,
    SARGConfig
)
from data_utils import convert_tags, data_iter
from transformers import AdamW, get_linear_schedule_with_warmup


logger = logging.getLogger(__name__)


TAGS = {
    "DELETE": 0,
    "KEEP": 1,
    "CHANGE": 2
}


def get_examples(examples_path, tokenizer: BertTokenizer, mode, max_ctx_len=80, max_src_len=50, max_add_len=15, language="zh"):
    assert mode in ["wechat", "ailab", "canard"]

    directory, filename = os.path.split(examples_path)
    cached_features_file = os.path.join(
        directory, f"cached_ctx_{max_ctx_len}_src_{max_src_len}_add_{max_add_len}_" + filename + ".pkl"
    )
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        with open(cached_features_file, "rb") as handle:
            examples = pickle.load(handle)
    else:
        examples = {
            "ctx_token": [],
            "ctx_mask": [],
            "ctx_pos": [],
            "ctx_ute": [],

            "src_token": [],
            "src_mask": [],
            "src_pos": [],
            "src_ute": [],

            "target": []
        }

        for contexts, source, target in tqdm(data_iter(examples_path, mode)):
            ctx = [tokenizer.cls_token_id]
            ctx_ute = []
            for u in range(len(contexts)):
                ui = len(contexts) - u + 1
                cu = tokenizer.tokenize(contexts[u])
                ctx.extend(tokenizer.convert_tokens_to_ids(cu) + [tokenizer.sep_token_id])
                ctx_ute.extend([ui] * (len(ctx) - len(ctx_ute)))

            if len(ctx) > max_ctx_len:
                ctx = ctx[-max_ctx_len:]
                ctx_ute = ctx_ute[-max_ctx_len:]
                if ctx[0] != tokenizer.cls_token_id:
                    ctx[0] = tokenizer.cls_token_id

            ctx_mask = [1] * len(ctx)
            ctx_pos = [x for x in range(len(ctx))]

            tags, src = convert_tags(source, target, tokenizer, debug=False, en=(language=="en"))
            src = tokenizer.convert_tokens_to_ids(src)[:max_src_len]
            tags = tags[:max_src_len]
            src_mask = [1] * len(src)
            src_pos = [x for x in range(len(src))]
            src_ute = [1] * len(src)

            target = [[] for _ in range(len(src))]
            for idx in range(len(src)):
                if tags[idx].startswith("CHANGE|"):
                    add = ["[CLS]"] + tags[idx][len("CHANGE|"):].split("<|>")
                    add = tokenizer.convert_tokens_to_ids(add)[:max_add_len-1] + [102]

                    target[idx].extend([TAGS["CHANGE"]] + add)
                else:
                    target[idx].append(TAGS[tags[idx]])

            examples["ctx_token"].append(ctx)
            examples["ctx_mask"].append(ctx_mask)
            examples["ctx_pos"].append(ctx_pos)
            examples["ctx_ute"].append(ctx_ute)

            examples["src_token"].append(src)
            examples["src_mask"].append(src_mask)
            examples["src_pos"].append(src_pos)
            examples["src_ute"].append(src_ute)

            examples["target"].append(target)

        logger.info(f"There are {len(examples['src_token'])} examples in {filename}")
        with open(cached_features_file, "wb") as handle:
            pickle.dump(examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return examples


class ExampleDataset(Dataset):
    def __init__(self, dict_examples):
        self.examples = dict_examples

    def __len__(self):
        return len(list(self.examples.values())[0])

    def __getitem__(self, i):
        single_example = {}
        for k, v in self.examples.items():
            single_example.update({k: v[i]})
        return single_example


class DictDataCollator(DataCollator):
    @overrides
    def collate_batch(self, batch_dict_examples):
        inputs = {}

        ctx_token = []
        ctx_mask = []
        ctx_pos = []
        ctx_ute = []

        src_token = []
        src_mask = []
        src_pos = []
        src_ute = []

        target = []

        ctx_max_len = -1
        src_max_len = -1
        add_max_len = -1

        for x in batch_dict_examples:
            ctx_max_len = max(len(x["ctx_token"]), ctx_max_len)
            src_max_len = max(len(x["src_token"]), src_max_len)
            for t in x["target"]:
                add_max_len = max(add_max_len, len(t))

        for dict_example in batch_dict_examples:
            ctx_token.append(dict_example["ctx_token"] + [0] * (ctx_max_len - len(dict_example["ctx_token"])))
            ctx_mask.append(dict_example["ctx_mask"] + [0] * (ctx_max_len - len(dict_example["ctx_token"])))
            ctx_pos.append(dict_example["ctx_pos"] + [p for p in range(len(dict_example["ctx_pos"]), ctx_max_len)])
            ctx_ute.append(dict_example["ctx_ute"] + [0] * (ctx_max_len - len(dict_example["ctx_token"])))

            src_token.append(dict_example["src_token"] + [0] * (src_max_len - len(dict_example["src_token"])))
            src_mask.append(dict_example["src_mask"] + [0] * (src_max_len - len(dict_example["src_token"])))
            src_pos.append(dict_example["src_pos"] + [p for p in range(len(dict_example["src_pos"]), src_max_len)])
            src_ute.append(dict_example["src_ute"] + [1] * (src_max_len - len(dict_example["src_token"])))

            _tgt = []
            for t in dict_example["target"]:
                _tgt.append(t + [0] * (add_max_len - len(t)))
            target.append(_tgt + [[0] * add_max_len] * (src_max_len - len(dict_example["src_token"])))

        inputs["ctx_token"] = torch.tensor(ctx_token, dtype=torch.long)
        inputs["ctx_mask"] = torch.tensor(ctx_mask, dtype=torch.long)
        inputs["ctx_pos"] = torch.tensor(ctx_pos, dtype=torch.long)
        inputs["ctx_ute"] = torch.tensor(ctx_ute, dtype=torch.long)

        inputs["src_token"] = torch.tensor(src_token, dtype=torch.long)
        inputs["src_mask"] = torch.tensor(src_mask, dtype=torch.long)
        inputs["src_pos"] = torch.tensor(src_pos, dtype=torch.long)
        inputs["src_ute"] = torch.tensor(src_ute, dtype=torch.long)

        inputs["target"] = torch.tensor(target, dtype=torch.long)

        return inputs


@dataclass
class ModelArguments:

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    cov_weight: Optional[float] = field(
        default=0., metadata={"help": "Used to control the cov loss"}
    )
    blend_gen: int = field(
        default=1, metadata={"help": "use vocab generation or not"}
    )
    blend_copy: int = field(
        default=1, metadata={"help": "use copy or not"}
    )
    mix_neighbors: int = field(
        default=0, metadata={"help": "mix the dummy tokens with their neighbors or not"}
    )
    change_weight: float = field(
        default=1.0, metadata={"help": "the loss weight of change operation"}
    )
    alpha: float = field(
        default=1.0, metadata={"help": "the alpha of tag loss"}
    )


@dataclass
class DataTrainingArguments:

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    append_ooc_file: Optional[str] = field(
        default=None,
        metadata={"help": "out of contexts file path"},
    )
    max_src_len: Optional[int] = field(
        default=50,
        metadata={"help": "The length to truncate your src input"}
    )
    max_ctx_len: Optional[int] = field(
        default=80,
        metadata={"help": "The length to truncate your ctx input"}
    )
    max_add_len: Optional[int] = field(
        default=15,
        metadata={"help": "The length to truncate your truncated added text"},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    mode: str = field(
        default=None, metadata={"help": "Set mode to 'ailab', 'wechat', 'canard'"}
    )
    train_ratio: float = field(
        default=0.9, metadata={"help": "The ratio to split your train and eval sets"}
    )
    patience: Optional[int] = field(
        default=10000, metadata={"help": "How many steps for tolerating the unimproved case"}
    )
    output_best_dir: str = field(
        default=None, metadata={"help": "The best model's output dir"}
    )
    language: str = field(
            default="zh", metadata={"help": "language for chinese(zh) or english(en)"}
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. try set overwrite_cache"
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    if model_args.config_name:
        config = SARGConfig.from_pretrained(model_args.config_name)
    elif model_args.model_name_or_path:
        config = SARGConfig.from_pretrained(model_args.model_name_or_path)
    else:
        config = SARGConfig()
        logger.warning("You are instantiating a new config instance from scratch.")

        logger.info(f"Set the blend_gen to {bool(model_args.blend_gen)}")
        config.blend_gen = bool(model_args.blend_gen)

        logger.info(f"Set the blend_copy to {bool(model_args.blend_copy)}")
        config.blend_copy = bool(model_args.blend_copy)

        logger.info(f"Set the mix_neighbors to {bool(model_args.mix_neighbors)}")
        config.mix_neighbors = bool(model_args.mix_neighbors)

        logger.info(f"Set the add_entity to {bool(model_args.add_entity) or bool(bool(model_args.add_cut))}")
        config.add_entity = bool(model_args.add_entity) or bool(bool(model_args.add_cut))

        logger.info(f"Set the change_weight to {model_args.change_weight}")
        config.change_weight = model_args.change_weight

        logger.info(f"Set the alpha to {model_args.alpha}")
        config.alpha = model_args.alpha

    if model_args.tokenizer_name:
        tokenizer = BertTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
           "set the tokenizer_name or model_name_or_path"
        )

    if model_args.model_name_or_path:
        logger.info(f"Set the blend_gen to {bool(model_args.blend_gen)}")
        logger.info(f"Set the mix_neighbors to {bool(model_args.mix_neighbors)}")
        logger.info(f"Set the add_entity to {bool(model_args.add_entity) or bool(bool(model_args.add_cut))}")
        logger.info(f"Set the blend_copy to {bool(model_args.blend_copy)}")
        logger.info(f"Set the change_weight to {model_args.change_weight}")
        logger.info(f"Set the alpha to {model_args.alpha}")
        model = SARGModel.from_pretrained(model_args.model_name_or_path,
                                          mix_neighbors=bool(model_args.mix_neighbors),
                                          blend_gen=bool(model_args.blend_gen),
                                          blend_copy=bool(model_args.blend_copy),
                                          add_entity=bool(model_args.add_entity) or bool(bool(model_args.add_cut)),
                                          change_weight=model_args.change_weight,
                                          alpha=model_args.alpha)
    else:
        logger.info("Training new model from scratch")
        model = SARGModel(config)

    # revise cov weight
    logger.info(f"Set the cov weight to {model_args.cov_weight}")
    model.config.cov_weight = model_args.cov_weight

    train_examples = get_examples(examples_path=data_args.train_data_file,
                                  tokenizer=tokenizer,
                                  max_src_len=data_args.max_src_len,
                                  max_ctx_len=data_args.max_ctx_len,
                                  max_add_len=data_args.max_add_len,
                                  mode=data_args.mode,
                                  language=data_args.language)
    eval_examples = get_examples(examples_path=data_args.eval_data_file,
                                 tokenizer=tokenizer,
                                 max_src_len=data_args.max_src_len,
                                 max_ctx_len=data_args.max_ctx_len,
                                 max_add_len=data_args.max_add_len,
                                 mode=data_args.mode,
                                 language=data_args.language
                                 )

    train_dataset = ExampleDataset(train_examples)
    eval_dataset = ExampleDataset(eval_examples)
    data_collator = DictDataCollator()

    t_total = int(len(train_dataset) // training_args.gradient_accumulation_steps * 30)  # fix decay scheduler

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate, eps=training_args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=t_total
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
        patience=data_args.patience,
        output_best_dir=data_args.output_best_dir,
        optimizers=(optimizer, scheduler)
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    return


if __name__ == "__main__":
    main()
