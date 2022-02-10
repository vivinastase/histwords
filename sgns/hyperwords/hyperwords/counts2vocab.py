from collections import Counter

from docopt import docopt

from representations.matrix_serializer import save_count_vocabulary

import ioutils

def main():
    args = docopt("""
    Usage:
        counts2pmi.py <counts>
    """)
    
    counts_path = args['<counts>']

    words = Counter()
    contexts = Counter()
    with open(counts_path) as f:
        for line in f:
            count, word, context = line.strip().split()
            count = int(count)
            words[word] += count
            contexts[context] += count

    words = sorted(words.items(), key=lambda item : item[1], reverse=True)
    contexts = sorted(contexts.items(), key=lambda item : item[1], reverse=True)

    save_count_vocabulary(counts_path + '.words.vocab', words)
    ioutils.write_pickle(words, counts_path + ".words.vocab.pkl")
    
    save_count_vocabulary(counts_path + '.contexts.vocab', contexts)
    ioutils.write_pickle(contexts, counts_path + ".contexts.vocab.pkl")


if __name__ == '__main__':
    main()
