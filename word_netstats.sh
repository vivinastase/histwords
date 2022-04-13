#!/bin/sh


### Example calls:
### ./word_netstats.sh -c /home/vivi/work/Projects/HistoricalWordAnalysis/Data/child_speech_corpus/lemmatized -d /home/vivi/work/Projects/HistoricalWordAnalysis/Data/child_speech_corpus/embeddings/lemmatized/ -o ../Outputs/child_speech_lematized -s 1 -l 3 -i 1

CORPUS_DIR=""
DATA_DIR=""
OUTPUT_DIR="./"
WORDS_FILE=""

START_YEAR=1
LAST_YEAR=3
YEAR_INC=1

# Parse input params

while [ $# -gt 0 ]; do
  case $1 in
    -c|--corpus_dir)
      CORPUS_DIR="$2"   ## directory with time-sliced corpus
      shift # past argument
      shift # past value
      ;;
    -d|--data_dir)
      DATA_DIR="$2"  ## directory with processed information about the corpus
      shift # past argument
      shift # past value
      ;;
    -o|--output_dir)
      OUTPUT_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    -s|--start_year)
      START_YEAR="$2"
      shift # past argument
      shift # past value
      ;;
    -l|--last_year)
      LAST_YEAR="$2"
      shift # past argument
      shift # past value
      ;;
    -i|--inc_year)
      YEAR_INC="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done


if [ -z $CORPUS_DIR ]; then
   echo "Path to a directory with the time_sliced corpus must be provided."
   echo "exiting"
   exit
fi

if [ -z $DATA_DIR ]; then
   echo "Path to a directory with the processed information about the time slices must be provided."
   echo "exiting"
   exit
fi



WORDS_FILE=${DATA_DIR}/${START_YEAR}-index.pkl

HIST_HOME='/home/vivi/work/Projects/HistoricalWordAnalysis/histwords-master'
SCRIPTS_PATH='statutils'
INDEX_SCRIPTS_PATH='googlengram/pullscripts'
export PYTHONPATH=${PYTHONPATH}:${HIST_HOME}:${HIST_HOME}/${SCRIPTS_PATH}:${HIST_HOME}/${INDEX_SCRIPTS_PATH}

echo "Making index files"
python3.8 ${INDEX_SCRIPTS_PATH}/gramgrab_notGN.py -c $CORPUS_DIR -s 2 -n 4 -o $DATA_DIR


## if I make edits to the sparse_io.pyx module, to recompile the file the following line may be needed (or adjusted, depending on where numpy is)
#export CFLAGS=-I/home/vivi/anaconda3/envs/histwords/lib/python3.8/site-packages/numpy/core/include/

echo "Computing semantic displacement for words in $WORDS_FILE"
python3.8 ${SCRIPTS_PATH}/netstats.py --start-year $START_YEAR --end-year $LAST_YEAR --year-inc $YEAR_INC --thresh 3 $DATA_DIR $WORDS_FILE 4 $OUTPUT_DIR


