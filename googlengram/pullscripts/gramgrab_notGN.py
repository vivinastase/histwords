'''
Minimally edited gramgrab to generate indices for data other than the google ngrams (which contain POS tags and have fixed length)
'''

import re
import os
import collections
import argparse
from multiprocessing import Process, Queue
import random
import time

#import sys
#sys.path.append("/home/vivi/work/Projects/HistoricalWordAnalysis/histwords-master")


import ioutils
from googlengram import indexing

import pyximport
pyximport.install()
from representations import sparse_io

LINE_SPLIT = 100000000

def update_count(ngram, target_ind, year, count, year_counters):
    item_id = ngram[target_ind]
    for i, context_id in enumerate(ngram):
        if i == target_ind:
            continue
        pair = (item_id, context_id)
        year_counters[year][pair] += count
    return year_counters


def make_ngrams(wordlist, n):
    ngrams = []
    for i in range(len(wordlist)-n-1):
        ngrams.append(wordlist[i:i+n])        
    return ngrams


def main(proc_num, queue, out_dir, download_dir, context_size):
    print(proc_num, "Start loop")
    while True:
        if queue.empty():
            break
        name = queue.get()
        year = name
        loc_dir = out_dir + "/" + name
        #ioutils.mkdir(loc_dir)

        print(proc_num, "Going through", name)
        index = collections.OrderedDict()
        year_counters = collections.defaultdict(collections.Counter)
        time.sleep(120 * random.random())
        with open(download_dir + name) as f:
            for i, l in enumerate(f):
                split = make_ngrams(l.strip().split(' '),5)
                
                #print("line {} ({}) -- ngrams in year {}: {}".format(i, f, year, split))
                
                for seq in split:
                    ngram = [indexing.word_to_id(word, index) for word in seq]
                    count = 1
                    if context_size == 2:
                        year_counters = update_count(ngram, 2, year, count, year_counters)
                    elif context_size == 4:
                        year_counters = update_count(ngram, 0, year, count, year_counters)
                        year_counters = update_count(ngram, 4, year, count, year_counters)
                    else:
                        raise Exception("Unsupported context size")

        print(proc_num, "Writing", name,"to file",loc_dir)
        time.sleep(120 * random.random())
        #print("\n_______________________________\nyear counters for year {}:\n{}\n_______________________________\n".format(year, year_counters[year]))
        sparse_io.export_mat_from_dict(year_counters[year], (loc_dir+".bin").encode('utf-8'))
        
        print("Writing pickle {}".format(loc_dir + "-index.pkl"))
        ioutils.write_pickle(index, loc_dir + "-index.pkl")


def run_parallel(num_processes, root_dir, out_dir, context_size):
    queue = Queue()
    download_dir = root_dir 
    #out_dir += '/c' + str(context_size) 
    ioutils.mkdir(out_dir)

    for name in os.listdir(download_dir):
        queue.put(name)
    procs = [Process(target=main, args=[i, queue, out_dir, download_dir, context_size]) for i in range(num_processes)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parses 5gram data into co-occurrence matrices")
    parser.add_argument("-c", "--corpus_dir", help="root directory where data lives")
    parser.add_argument("-s", "--context_size", type=int, help="Size of context window. Currently only size 2 and 4 are supported.")
    parser.add_argument("-n", "--num_procs", type=int, help="number of processes to spawn")
    parser.add_argument("-o", "--out_dir", type=str, help="output directory")
    args = parser.parse_args()
    run_parallel(args.num_procs, args.corpus_dir, args.out_dir, args.context_size) 
