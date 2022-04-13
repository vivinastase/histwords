'''
Created on Feb 21, 2022

@author: vivi
'''

import os
import time
import random
import argparse
import re

import scipy

from multiprocessing import Process, Queue

import ioutils

from viz.common import load_embeddings



def get_distance(a, b, dist):
    
    if "cos" in dist:
        return scipy.spatial.distance.cosine(a,b)
    if "euclid" in dist:
        return scipy.spatial.distance.euclidean(a, b)
    
    return a.dot(b.T)

def get_polysemy(embeds, words, type):
    polysemy = {}
    print(embeds.m.shape)
    for word in words:
        if embeds.oov(word):
            polysemy[word] = float('nan')
        else:
            polysemy[word] = get_distance(base_embeds.represent(word), delta_embeds.represent(word), dist)
    return polysemy


def merge(out_pref, years, word_list):
    polysemy_yearstats = {}
    for word in word_list:
        polysemy_yearstats[word] = {}
    for year in years:
        polysemy_yearstat = ioutils.load_pickle(out_pref + str(year) + "-polysemy.pkl")
        for word in word_list:
            if word not in polysemy_yearstat:
                polysemy = float('nan')
            else:
                polysemy = polysemy_yearstat[word]
            polysemy_yearstats[word][year] = polysemy
        os.remove(out_pref + str(year) + "-polysemy.pkl")
    ioutils.write_pickle(disp_yearstats, out_pref + "polysemy.pkl")

    txt_out = open(out_pref + "/polysemy_info.txt", "w")
    txt_out.write("word\t" + "\t".join([str(y) for y in years])+ "\n")
    for word in sorted(polysemy_yearstats):
        str_out = word
        for year in years:
            str_out += "\t" + str(polysemy_yearstats[word][year])
        if not re.search(r"\tnan",str_out):
            txt_out.write(str_out + "\n")
        
    txt_out.close()
    
    

def worker(proc_num, queue, out_pref, in_dir, target_lists, context_lists, embeddings, thresh, type):
    time.sleep(10*random.random())
    while True:
        if queue.empty():
            print(proc_num, "Finished")
            break
        year = queue.get()
        print(proc_num, "Getting deltas...")
        year_polysemy = get_polysemy(embeddings.get_embed(year), target_lists[year], type)
        print(proc_num, "Writing results...")
        ioutils.write_pickle(year_polysemy, out_pref + str(year) + "-polysemy.pkl")


def run_parallel(num_procs, out_pref, in_dir, years, target_lists, context_lists, embeddings, thresh, year_inc, disp_year, type, dist):
    queue = Queue()
    for year in years:
        queue.put(year)
    procs = [Process(target=worker, args=[i, queue, out_pref, in_dir, target_lists, context_lists, embeddings, thresh, type]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print("Merging")
    full_word_set = set([])
    for year_words in target_lists.values():
        full_word_set = full_word_set.union(set(year_words))
    merge(out_pref, years, list(full_word_set))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Computes semantic change statistics for words.")
    parser.add_argument("dir", help="path to embeddings")
    parser.add_argument("num_procs", type=int, help="number of processes to spawn")
    parser.add_argument("word_file", help="path to sorted word file")
    parser.add_argument("out_dir", help="output path")
    parser.add_argument("--target-words", type=int, help="Number of words (of decreasing average frequency) to analyze", default=-1)
    parser.add_argument("--context-words", type=int, help="Number of words (of decreasing average frequency) to include in context. -2 means all regardless of word list", default=-1)
    parser.add_argument("--context-word-file")
    parser.add_argument("--start-year", type=int, help="start year (inclusive)", default=1800)
    parser.add_argument("--year-inc", type=int, help="year increment", default=10)
    parser.add_argument("--end-year", type=int, help="end year (inclusive)", default=2000)
    args = parser.parse_args()
    
    years = range(args.start_year, args.end_year + 1, args.year_inc)
    
    target_lists, context_lists = ioutils.load_target_context_words(years, args.word_file, args.target_words, -1)
    
    if args.context_word_file != None:
        print("Loading context words..")
        _ , context_lists = ioutils.load_target_context_words(years, args.word_file, -1, args.context_words)
        
    target_lists, context_lists = ioutils.load_target_context_words(years, args.word_file, args.target_words, args.context_words)
    
    out_pref = args.out_dir + "/" + args.dist + "/"
    ioutils.mkdir(out_pref)
    print("Writing results to {}".format(out_pref))
    
    
    run_parallel(args.num_procs, out_pref, args.dir + "/", years[1:], target_lists, context_lists, load_embeddings(args.dir))       
