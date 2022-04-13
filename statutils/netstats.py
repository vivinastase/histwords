import random
import os
import time
import argparse
from queue import Empty
from multiprocessing import Process, Queue

import sys
sys.path.append("/home/vivi/work/Projects/HistoricalWordAnalysis/histwords-master")

import ioutils

import pyximport
pyximport.install()
from representations import sparse_io

from googlengram.indexing import get_full_word_list

import numpy as np

import pandas as pd
import math

"""
Get statistics on PPMI network such as degrees and clustering coefficients.
"""

NAN = float('nan')
STATS = ["deg", "sum", "bclust", "wclust"]

def compute_word_stats(mat, word, year, word_index, one_word, index_set = None, stats=STATS, out_pref="./"):
    if not word in word_index:
        return {stat:NAN for stat in stats}
    word_i = word_index[word] 
    if index_set != None and not word_i in index_set:
        return {stat:NAN for stat in stats}
    if word_i >= mat.shape[0]: 
        return {stat:NAN for stat in stats}
    vec = mat[word_i, :]
    indices = vec.nonzero()[1]
    if len(indices) > 1:
        print("{} ==> non-zero indices, with max = {}".format(word, max(indices)))
    else:
        print("{} ==> empty indices".format(word))
    vec = vec[:, indices]
    # only compute clustering if we have too..
    if "bclust" in stats or "wclust" in stats:
        if len(indices) >= 2:
            weights = vec/vec.sum()
            reduced = mat[indices, :]
            reduced = reduced[:, indices]
            reduced.eliminate_zeros()
            weighted = (weights * reduced).sum() / (float(len(indices)) - 1)
            binary = float(reduced.nnz) / (len(indices) * (len(indices) - 1)) 
        else:
            weighted = binary = 0
    deg = len(indices)
    sum = vec.sum()
    vals = {}
    if "deg" in stats:
        vals["deg"] = deg
    if "sum" in stats:
        vals["sum"] = sum
    if "bclust" in stats:
         vals["bclust"] = binary
    if "wclust" in stats:
         vals["wclust"] = weighted
         
    if word in one_word.split(" "):
        write_details(word, year, word_index, indices, mat, out_pref)
         
    return vals



def write_details(word, year, word_index, indices, mat, out_pref):
    
    ## sort the indices so in the output cooccurrences and matrix everything is ordered    
    indices.sort()
    
    print("{} (id = {} in year = {}) => coocs = {}".format(word, word_index[word], year, [w for (w, id) in word_index.items() if id in indices]))

    coocs = pd.DataFrame({(id, w) for (w, id) in word_index.items() if id in indices})
    coocs.to_csv(out_pref + str(year) + "_" + word + "_coocs.txt", ",")
    
    reduced = mat[indices, :]
    reduced = reduced[:, indices]
    reduced_df = pd.DataFrame(np.matrix(reduced.toarray()))
    reduced_df.to_csv(out_pref + str(year) + "_" + word + "_coocs_matrix.txt", ",")



def get_year_stats(mat, year, year_index, word_list, one_word, index_set=None, stats=STATS, out_pref = "./"):
    mat.setdiag(0)
    mat = mat.tocsr()
    year_stats = {stat:{} for stat in stats}
    for i, word in enumerate(word_list):
        single_word_stats = compute_word_stats(mat, word, year, year_index, one_word, index_set=index_set, stats=["sum", "deg", "bclust", "wclust"], out_pref=out_pref)
        if i % 1000 == 0:
            print("Done ", i)
        for stat in single_word_stats:
            year_stats[stat][word] = single_word_stats[stat]
    return year_stats



def merge(out_pref, years, full_word_list):
    merged_word_stats = {}
    for stat in STATS:
        merged_word_stats[stat] = {}
        for word in full_word_list:
            merged_word_stats[stat][word] = {}
    for year in years:
        year_stats = ioutils.load_pickle(out_pref + str(year) + "-tmp.pkl")
        for stat, stat_vals in year_stats.items():
            for word in full_word_list:
                if not word in stat_vals:
                    merged_word_stats[stat][word][year] = NAN
                else:
                    merged_word_stats[stat][word][year] = stat_vals[word]
        os.remove(out_pref + str(year) + "-tmp.pkl")
    ioutils.write_pickle(merged_word_stats, out_pref +  "/netstats.pkl")


    ## write info to csv file
    word_stats_df = pd.DataFrame([(stat,word,year,val) for stat,wordyearval in merged_word_stats.items()
                                                        for word,yearval in wordyearval.items()
                                                         for year,val in yearval.items()])
    word_stats_df.columns = ["stat", "word", "year", "val"]
    word_stats_df.sort_values(["word", "year", "stat"], inplace=True)   
    word_stats_df.to_csv(out_pref + "/netstats.csv", sep="\t")
    
    write_diffs(merged_word_stats, years, ["bclust", "wclust"], out_pref)


def write_diffs(word_stats, years, stats, out_pref):
    
    years = sorted(years)
    
    diffs = {}
    for stat in stats:
        diffs[stat] = {}
        for word in word_stats[stat]:
            diffs[stat][word] = {}
            for i in range(1,len(years)):
                y1 = years[i-1]
                y2 = years[i]
                if has_val(y1,word_stats[stat][word]) and has_val(y2,word_stats[stat][word]):
                    diffs[stat][word][str(y2) + "-" + str(y1)] = word_stats[stat][word][y2]-word_stats[stat][word][y1]
                    #print("{}: *{}*\t{}: *{}*".format(y2,word_stats[stat][word][y2],y1,word_stats[stat][word][y1]))
                    
                    
    stat_diffs_df = pd.DataFrame([(stat, word, year, val) for stat,wordyearval in diffs.items()
                                                            for word,yearval in wordyearval.items()
                                                             for year,val in yearval.items() ])
    
    stat_diffs_df.columns = ["stat", "word", "years", "val"]
    #print("stat diffs: \n{}\n".format(stat_diffs_df))
    stat_diffs_df.sort_values(["stat", "years", "val", "word"], inplace=True)
    stat_diffs_df.to_csv(out_pref + "/cluster_diffs.csv", sep="\t")
        
    
def has_val(x, d):
    return (x in d.keys()) and isinstance(d[x], (int, float)) and (not math.isnan(d[x]))
    
    
    

def worker(proc_num, queue, out_pref, in_dir, year_index_infos, thresh, one_word):
    print(proc_num, "Start loop")
    time.sleep(10 * random.random())
    while True:
        try: 
            year = queue.get(block=False)
        except Empty:
            print(proc_num, "Finished")
            break

        print(proc_num, "Retrieving mat for year", year)
        filename = (in_dir + str(year) + ".bin").encode("utf-8")
        
        if thresh != None:
            mat = sparse_io.retrieve_mat_as_coo_thresh(filename, thresh)
        else:
            mat = sparse_io.retrieve_mat_as_coo(filename, min_size=5000000)
            
        print("loaded matrix with dims {}".format(mat.shape))
        print(proc_num, "Getting stats for year", year)
        year_stats = get_year_stats(mat, year, year_index_infos[year]["index"], year_index_infos[year]["list"], one_word, index_set = set(year_index_infos[year]["indices"]), out_pref = out_pref)

        print(proc_num, "Writing stats for year", year)
        ioutils.write_pickle(year_stats, out_pref + str(year) + "-tmp.pkl")
        
        

def run_parallel(num_procs, out_pref, in_dir, year_index_infos, thresh, one_word):
    queue = Queue()
    years = list(year_index_infos.keys())
    
    print("Years: {}".format(years))
    
    random.shuffle(years)
    for year in years:
        queue.put(year)
    procs = [Process(target=worker, args=[i, queue, out_pref, in_dir, year_index_infos, thresh, one_word]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print("Merging")
    merge(out_pref, years, get_full_word_list(year_index_infos))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Computes network-based statistics for each word.")
    parser.add_argument("dir", help="path to directory with nppmi data and year indexes")
    parser.add_argument("word_file", help="path to sorted word file(s).", default=None)
    parser.add_argument("num_procs", type=int, help="number of processes to spawn")
    parser.add_argument("outdir", help="path to directory with nppmi data and year indexes")

    parser.add_argument("--num-words", type=int, help="Number of words (of decreasing average frequency) to include. Must also specifiy word file and index.", default=-1)
    parser.add_argument("--min-count", type=int, help="Min word count for inclusion in network. Must also include count dir if this used.", default=-1)
    parser.add_argument("--count-dir", type=int, help="Directory with count data.", default=-1)
    parser.add_argument("--start-year", type=int, help="start year (inclusive)", default=1900)
    parser.add_argument("--end-year", type=int, help="start year (inclusive)", default=2000)
    parser.add_argument("--year-inc", type=int, help="year increment", default=1)
    parser.add_argument("--thresh", type=float, help="optional threshold", default=None)
    
    parser.add_argument("--word", default="dreymdi drykk herra munn skyldum handar veislu", help="a word for which to generate cooccurrence word list and matric files, to produce then visualizations")
    
    args = parser.parse_args()
    years = range(args.start_year, args.end_year + 1, args.year_inc)
    year_index_infos = ioutils.load_year_index_infos(args.dir, years, args.word_file, num_words=args.num_words)
    
    '''
    outpref ="/netstats/" + args.word_file.split("/")[-1].split(".")[0]
    if args.num_words != -1:
        outpref += "-top" + str(args.num_words)
    if args.thresh != None:
        outpref += "-" + str(args.thresh)
    ioutils.mkdir(args.dir + "/netstats")
    '''
    
    outpref = "/netstats/"
    ioutils.mkdir(args.outdir + outpref)
    run_parallel(args.num_procs, args.outdir + outpref, args.dir + "/", year_index_infos, args.thresh, args.word)       
