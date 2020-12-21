import os
import json
import torch
from transformers import BertTokenizer
from modeling_sarg import SARGModel
from tqdm import tqdm
from run_train import get_examples, ExampleDataset, DictDataCollator
from torch.utils.data import DataLoader
from data_utils import convert_tokens_to_string, insert_dummy, _decode_valid_tags
from score import Scorer, BasicTokenizer


TAGS = {
    0: "DELETE",
    1: "KEEP",
    2: "CHANGE"
}

basic_tokenizer = BasicTokenizer()  # just for evaluation


def eval_on_standard_test_set(
        test_path, 
        model_path, 
        tokenizer_path, 
        mode, 
        batch_size, 
        add_len, 
        language, 
        forbid_repeat,
        max_src_len,
        max_ctx_len,
        device=None, 
        **kwargs):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if forbid_repeat and language == 'en':
        print('Recommend that set the forbid_repeat to false when eval on the CANARD.')
    n_beams = kwargs.get('n_beams', 1)
    directory = model_path
    dump_rlt_path = os.path.join(directory, mode + f'_n_beams_{n_beams}.metrics.json')

    model = SARGModel.from_pretrained(model_path).to(device)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    predictions = []
    references = []
    orig_utterances = []

    pos_predictions = []
    pos_references = []
    neg_predictions = []
    neg_references = []

    examples = get_examples(
        examples_path=test_path,
        mode=mode,
        tokenizer=tokenizer,
        max_add_len=add_len,
        max_src_len=max_src_len,
        max_ctx_len=max_ctx_len
    )
    test_set = ExampleDataset(examples)
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=DictDataCollator().collate_batch)

    # use raw data to speed up evaluating
    with open(test_path, 'r', encoding='utf-8') as f:
        st = 0
        raw_data = [x.strip() for x in f]

    for x in tqdm(test_loader):
        bsz = x['src_token'].size(0)
        lines = raw_data[st: st + bsz]
        st += bsz

        for k, v in x.items():
            if not k.startswith("target"):
                x[k] = v.to(device)
        rlt = model.infer(x, add_len, **kwargs)
        tags = rlt['tag'].tolist()  # (batch_size, src_len)
        bidx_sidx_to_idx = rlt.get('bidx_sidx_to_idx', dict())
        preds = rlt.get('preds', [[]] * rlt['tag'].shape[0])  # (batch_size, src_len, add_len)

        for i in range(rlt['tag'].shape[0]):
            for j in range(rlt['tag'].shape[1]):
                tags[i][j] = TAGS[tags[i][j]]

        # revise tags
        pred_set = set()
        for bidx_sidx, idx in bidx_sidx_to_idx.items():
            bidx, sidx = bidx_sidx
            assert tags[bidx][sidx] == "CHANGE"
            pred = tokenizer.decode(preds[idx], clean_up_tokenization_spaces=True, skip_special_tokens=True).split()
            pred = convert_tokens_to_string(tokenizer, pred, en=language=="en")

            if forbid_repeat:
                if pred not in pred_set:
                    tags[bidx][sidx] += '|' + pred
                    pred_set.add(pred)
                else:
                    tags[bidx][sidx] += '|' + ''
            else:
                tags[bidx][sidx] += '|' + pred
                pred_set.add(pred)

        for tag, line in zip(tags, lines):

            if mode == 'wechat':
                line_split = line.split('\t\t')
                contexts, reference = line_split[:-1], line_split[-1]
            elif mode == "canard":
                line_split = line.split('\t')
                contexts, reference = line_split[:-1], line_split[-1]
            elif mode == 'ailab':
                line_split = line.split('\t')
                if line_split[-1] != '0':
                    contexts, reference = line_split[:5], line_split[-1]
                else:
                    contexts, reference = line_split[:5], line_split[4]
            else:
                raise ValueError('mode must be ailab, wechat or canard')
            source = insert_dummy(tokenizer.tokenize(contexts[-1]))
            prediction = ' '.join(basic_tokenizer.tokenize(_decode_valid_tags(source, tag, tokenizer, en=language=="en").replace('[UNK]', '')))
            predictions.append(prediction)
            references.append(' '.join(basic_tokenizer.tokenize(reference)))
            orig_utterances.append(' '.join(basic_tokenizer.tokenize(contexts[-1])))
            if contexts[-1] != reference:
                # pos
                pos_predictions.append(prediction)
                pos_references.append(' '.join(basic_tokenizer.tokenize(reference)))
            else:
                neg_predictions.append(prediction)
                neg_references.append(' '.join(basic_tokenizer.tokenize(reference)))

    # print("predictions:")
    # for ref, org, pred in zip(references, orig_utterances, predictions):
    #     print("ref:", ref)
    #     print("org:", org)
    #     print("pred:", pred)
    #     print("")

    scorer = Scorer()

    bleu_1, bleu_2, bleu_3, bleu_4 = scorer.corpus_bleu_score(references=references, predictions=predictions)
    em_all = scorer.em_score(references=references, predictions=predictions)
    em_pos = scorer.em_score(references=pos_references, predictions=pos_predictions)
    em_neg = scorer.em_score(references=neg_references, predictions=neg_predictions)
    rouge_1, rouge_2, rouge_l = scorer.rouge_score(references=references, predictions=predictions)
    p1, r1, f1 = scorer.resolution_score(xs=predictions, refs=references, oris=orig_utterances, ngram=1)
    p2, r2, f2 = scorer.resolution_score(xs=predictions, refs=references, oris=orig_utterances, ngram=2)
    p3, r3, f3 = scorer.resolution_score(xs=predictions, refs=references, oris=orig_utterances, ngram=3)

    metrics = {
        'bleu_1': bleu_1,
        'bleu_2': bleu_2,
        'bleu_3': bleu_3,
        'bleu_4': bleu_4,
        'rouge_1': rouge_1,
        'rouge_2': rouge_2,
        'rouge_l': rouge_l,
        'em_all': em_all,
        'em_pos': em_pos,
        'em_neg': em_neg,
        'p1_r1_f1': [p1, r1, f1],
        'p2_r2_f2': [p2, r2, f2],
        'p3_r3_f3': [p3, r3, f3],
    }

    with open(dump_rlt_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', default=None, type=str, help='The path of test data.')
    parser.add_argument('--model_path', default=None, type=str, help='The path of model')
    parser.add_argument('--tokenizer_path', default='BertTokenizer', help='The path of tokenizer')
    parser.add_argument('--mode', default='ailab', type=str, help='wechat, ailab or canard')
    parser.add_argument('--batch_size', default=1, type=int, help='The batch size of test loader')
    parser.add_argument('--n_beams', default=1, type=int, help='The size of Beam Search')
    parser.add_argument('--device', default=None, type=str, help='The device will be cpu or cuda')
    parser.add_argument('--forbid_repeat', default="True", type=str, help='whether to forbid the repeated add phrase')
    parser.add_argument('--language', default="zh", type=str, help='Language for chinese(zh) or english(en) ')
    parser.add_argument('--add_len', default=15, type=int, help='The length of added phrase')
    parser.add_argument('--max_ctx_len', default=80, type=int, help='Max length of context text')
    parser.add_argument('--max_src_len', default=50, type=int, help='Max length of source text')


    args = parser.parse_args()

    metrics = eval_on_standard_test_set(test_path=args.test_path,
                                        model_path=args.model_path,
                                        tokenizer_path=args.tokenizer_path,
                                        mode=args.mode,
                                        n_beams=args.n_beams,
                                        batch_size=args.batch_size,
                                        device=args.device,
                                        add_len=args.add_len,
                                        forbid_repeat=(args.forbid_repeat=="True"),
                                        language=args.language,
                                        max_ctx_len=args.max_ctx_len,
                                        max_src_len=args.max_src_len)
    print(metrics)
    return metrics


if __name__ == "__main__":
    main()

