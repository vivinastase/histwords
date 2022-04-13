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

def get_deltas(base_embeds, delta_embeds, words, type, dist):
    deltas = {}
    print(base_embeds.m.shape, delta_embeds.m.shape)
    for word in words:
        if base_embeds.oov(word) or delta_embeds.oov(word):
            deltas[word] = float('nan')
        else:
            #delta = base_embeds.represent(word).dot(delta_embeds.represent(word).T)
            #delta = scipy.spatial.distance.euclidean(base_embeds.represent(word), delta_embeds.represent(word))
            #delta = scipy.spatial.distance.cosine(base_embeds.represent(word), delta_embeds.represent(word))
            delta = get_distance(base_embeds.represent(word), delta_embeds.represent(word), dist)
            if type == "PPMI":
                delta = delta[0,0]
            deltas[word] = delta
    return deltas

def merge(out_pref, years, word_list):
    vol_yearstats = {}
    disp_yearstats = {}
    for word in word_list:
        vol_yearstats[word] = {}
        disp_yearstats[word] = {}
    for year in years:
        vol_yearstat = ioutils.load_pickle(out_pref + str(year) + "-vols.pkl")
        disp_yearstat = ioutils.load_pickle(out_pref + str(year) + "-disps.pkl")
        for word in word_list:
            if word not in vol_yearstat:
                vol = float('nan')
            else:
                vol = vol_yearstat[word]
            if word not in disp_yearstat:
                disp = float('nan')
            else:
                disp = disp_yearstat[word]
            vol_yearstats[word][year] = vol
            disp_yearstats[word][year] = disp
        os.remove(out_pref + str(year) + "-vols.pkl")
        os.remove(out_pref + str(year) + "-disps.pkl")
    ioutils.write_pickle(vol_yearstats, out_pref + "vols.pkl")
    ioutils.write_pickle(disp_yearstats, out_pref + "disps.pkl")




    word_stats_df = pd.DataFrame([(word,year,val)  for word,yearval in vol_yearstats.items()
                                                    for year,val in yearval.items()])
    word_stats_df.columns = ["word", "year", "val"]
    word_stats_df.sort_values(["word", "year"], inplace=True)       
    word_stats_df.to_csv(out_pref + "/displacement_info.csv", sep="\t")
    
    
    

def worker(proc_num, queue, out_pref, in_dir, target_lists, context_lists, embeddings, thresh, year_inc, disp_year, type, dist):
    time.sleep(10*random.random())
    disp_embedding = embeddings.get_embed(disp_year)
    while True:
        if queue.empty():
            print(proc_num, "Finished")
            break
        year = queue.get()
        print(proc_num, "Getting deltas...")
        year_vols = get_deltas(embeddings.get_embed(year-year_inc), embeddings.get_embed(year), target_lists[year], type, dist)
        year_disp = get_deltas(disp_embedding, embeddings.get_embed(year), target_lists[year], type, dist)
        print(proc_num, "Writing results...")
        ioutils.write_pickle(year_vols, out_pref + str(year) + "-vols.pkl")
        ioutils.write_pickle(year_disp, out_pref + str(year) + "-disps.pkl")

def run_parallel(num_procs, out_pref, in_dir, years, target_lists, context_lists, embeddings, thresh, year_inc, disp_year, type, dist):
    queue = Queue()
    for year in years:
        queue.put(year)
    procs = [Process(target=worker, args=[i, queue, out_pref, in_dir, target_lists, context_lists, embeddings, thresh, year_inc, disp_year, type, dist]) for i in range(num_procs)]
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
    parser.add_argument("--type", default="SVD")
    parser.add_argument("--end-year", type=int, help="end year (inclusive)", default=2000)
    parser.add_argument("--disp-year", type=int, help="year to measure displacement from", default=2000)
    parser.add_argument("--dist", type=str, default="cosine", help="distance metric")
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
    
    
    run_parallel(args.num_procs, out_pref, args.dir + "/", years[1:], target_lists, context_lists, load_embeddings(args.dir), 0, args.year_inc, args.disp_year, args.type, args.dist)       
