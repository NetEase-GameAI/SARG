def _compute_lcs(source, target):
    """Computes the Longest Common Subsequence (LCS).

  Description of the dynamic programming algorithm:
  https://www.algorithmist.com/index.php/Longest_Common_Subsequence

  Args:
    source: List of source tokens.
    target: List of target tokens.

  Returns:
    List of tokens in the LCS.
  """
    table = _lcs_table(source, target)
    return _backtrack(table, source, target, len(source), len(target))


def _lcs_table(source, target):
    """Returns the Longest Common Subsequence dynamic programming table."""
    rows = len(source)
    cols = len(target)
    lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if source[i - 1] == target[j - 1]:
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
            else:
                lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
    return lcs_table


def _backtrack(table, source, target, i, j):
    """Backtracks the Longest Common Subsequence table to reconstruct the LCS.

  Args:
    table: Precomputed LCS table.
    source: List of source tokens.
    target: List of target tokens.
    i: Current row index.
    j: Current column index.

  Returns:
    List of tokens corresponding to LCS.
  """
    if i == 0 or j == 0:
        return []
    if source[i - 1] == target[j - 1]:
        # Append the aligned token to output.
        return _backtrack(table, source, target, i - 1, j - 1) + [target[j - 1]]
    if table[i][j - 1] > table[i - 1][j]:
        return _backtrack(table, source, target, i, j - 1)
    else:
        return _backtrack(table, source, target, i - 1, j)


def insert_dummy(tokens, p='[unused%d]'):
    rlt = []
    cnt = 1
    for token in tokens:
        rlt.append(p % cnt)
        rlt.append(token)
        cnt += 1
    rlt.append(p % cnt)
    return rlt


def convert_tokens_to_string(tokenizer, tokens, en=False):
    if en:
        return tokenizer.convert_tokens_to_string(tokens)
    return ''.join(tokenizer.convert_tokens_to_string(tokens).split(' '))


def _decode_valid_tags(source, tags, tokenizer, en):
    string = []
    for token, tag in zip(source, tags):
        if tag == 'DELETE':
            continue
        elif tag == 'KEEP':
            string.append(token)
        else:
            string.append(tag.split('|')[-1])
    return convert_tokens_to_string(tokenizer, string, en)


def convert_tags(source, target, tokenizer, debug=False, en=False):

    source = insert_dummy(tokenizer.tokenize(source))
    target = tokenizer.tokenize(target)

    # initialize tags
    tags = ['DELETE'] * len(source)

    kept_tokens = _compute_lcs(source, target) + ['[DUMMY]']

    target_idx = 0
    phrase = []

    for source_idx in range(len(source)):
        if source[source_idx] == kept_tokens[0]:
            tags[source_idx] = 'KEEP'
            while target_idx < len(target) and target[target_idx] != kept_tokens[0]:
                phrase.append(target[target_idx])
                target_idx += 1

            kept_tokens = kept_tokens[1:]

            if len(phrase) > 0:
                if debug:
                    tags[source_idx - 1] = 'CHANGE|' + convert_tokens_to_string(tokenizer, phrase, en)
                else:
                    tags[source_idx - 1] = 'CHANGE|' + '<|>'.join(phrase)
                phrase = []

            target_idx += 1

    if target_idx < len(target):
        if debug:
            tags[-1] = 'CHANGE|' + convert_tokens_to_string(tokenizer, target[target_idx:], en)
        else:
            tags[-1] = 'CHANGE|' + "<|>".join(target[target_idx:])

    if debug and _decode_valid_tags(source, tags, tokenizer, en) != convert_tokens_to_string(tokenizer, target, en):
        print(f"decoded: {_decode_valid_tags(source, tags, tokenizer, en)} "
              f"original: {convert_tokens_to_string(tokenizer, target, en)}")
    return tags, source


def data_iter(file_path, mode):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if mode == 'wechat':
                line_split = line.split('\t\t')
                contexts_source, target = line_split[:-1], line_split[-1]
                contexts = contexts_source[:-1]
                source = contexts_source[-1]
            elif mode == "ailab":
                line_split = line.split('\t')
                if line_split[-1] != '0':
                    contexts_source, target = line_split[:5], line_split[-1]
                else:
                    contexts_source, target = line_split[:5], line_split[4]
                contexts = contexts_source[:-1]
                source = contexts_source[-1]
            elif mode == 'canard':
                line_split = line.split('\t')
                contexts_source, target = line_split[:-1], line_split[-1]
                contexts = contexts_source[:-1]
                source = contexts_source[-1]
            else:
                raise ValueError("mode must in [wechat, ailab, local]")
            yield contexts, source, target


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
