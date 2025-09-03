def build_parse_tree(table, words, symbol, start=0, length=None):
    if length is None:
        length = len(words)

    if length == 1:
        return [symbol, words[start]]

    for k in range(1, length):
        left_cell = table[k-1][start]
        right_cell = table[length-k-1][start+k]

        for rule in grammar:
            if grammar[rule] == [left_cell, right_cell]
                left_cell = build_parse_tree(table, words, left_cell[0], start, k)
                right_cell = build_parse_tree(table, words, right_cell[0], start+k, length-k)
