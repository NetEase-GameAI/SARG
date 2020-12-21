import unicodedata
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.
    Args:
      do_lower_case: Whether to lower case the input.
    """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)

        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


# s: sentence, each word is seperated by a spcace ' '
def make_gram(s, ngram = 1):
    words = s.split(' ')
    words = [word for word in words if len(word) > 0]
    grams = [' '.join(words[i:i + ngram]) for i in range(len(words) - ngram + 1)]
    return grams


class Scorer(object):

    @staticmethod
    def corpus_bleu_score(references, predictions):
        ref_list = [[ref.split(' ')] for ref in references]
        pred_list = [pred.split(' ') for pred in predictions]
        bleu1s = corpus_bleu(ref_list, pred_list, weights=(1.0, 0.0, 0.0, 0.0))
        bleu2s = corpus_bleu(ref_list, pred_list, weights=(0.5, 0.5, 0.0, 0.0))
        bleu3s = corpus_bleu(ref_list, pred_list, weights=(0.33, 0.33, 0.33, 0.0))
        bleu4s = corpus_bleu(ref_list, pred_list, weights=(0.25, 0.25, 0.25, 0.25))

        return bleu1s, bleu2s, bleu3s, bleu4s

    @staticmethod
    def em_score(references, predictions):
        if len(references) == 0:
            return 0
        matches = []
        for ref, cand in zip(references, predictions):
            if ref == cand:
                matches.append(1)
            else:
                matches.append(0)
        return sum(matches) / len(matches)

    @staticmethod
    def rouge_score(references, predictions):
        rouge = Rouge()
        for idx in range(len(predictions)):
            if predictions[idx].strip() == '':
                predictions[idx] = 'someword'
        scores = rouge.get_scores(hyps=predictions, refs=references, avg=True)
        rouge1s = scores['rouge-1']['f']
        rouge2s = scores['rouge-2']['f']
        rougels = scores['rouge-l']['f']
        return rouge1s, rouge2s, rougels

    @staticmethod
    def resolution_score(xs, refs, oris, ngram=1):
        """
        The code of calculating restoration-score is acquired from
           Pan et al (Improving open-domain dialogue systems via multi-turn incomplete utterance restoration)
       """
        p = []
        r = []
        for x, ref, ori in zip(xs, refs, oris):
            ori = set(make_gram(ori, ngram))
            x = set(make_gram(x, ngram)) - ori
            ref = set(make_gram(ref, ngram)) - ori
            for word in x:
                if word in ref:
                    p.append(1)
                else:
                    p.append(0)

            for word in ref:
                if word in x:
                    r.append(1)
                else:
                    r.append(0)
        p, r = np.mean(p), np.mean(r)
        return p, r, 2 * p * r / (p + r)

