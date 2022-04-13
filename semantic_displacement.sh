#!/bin/sh


### Example calls:
### ./semantic_displacement.sh -e embeddings/child_speech_lemmatized_svd/ -w embeddings/child_speech_lemmatized_svd/1-vocab.pkl -o ../Outputs/child_speech_lemmatized -d dot_prod
### ./semantic_displacement.sh -e embeddings/child_speech_lemmatized_svd/ -w embeddings/child_speech_lemmatized_svd/1-vocab.pkl -o ../Outputs/child_speech_lemmatized -d cosine
### ./semantic_displacement.sh -e embeddings/child_speech_lemmatized_svd/ -w embeddings/child_speech_lemmatized_svd/1-vocab.pkl -o ../Outputs/child_speech_lemmatized -d euclid


EMBS_DIR=""
OUTPUT_DIR="./"
WORDS_FILE=""

START_YEAR=1
LAST_YEAR=3
YEAR_INC=1

# Parse input params

while [ $# -gt 0 ]; do
  case $1 in
    -d|--dist)
      DIST="$2"   ## type of distance to compute (cosine, euclid, dotprod)
      shift # past argument
      shift # past value
      ;;
    -e|--embs_dir)
      EMBS_DIR="$2"
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
    -w|--words)
      WORDS_FILE="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done



if [ -z $EMBS_DIR ]; then
   echo "Path to an embeddings directory must be provided."
   echo "exiting"
   exit
fi

if [ -z $WORDS_FILE ]; then
   echo "Words file must be provided."
   echo "exiting"
   exit
fi



HIST_HOME='/home/vivi/work/Projects/HistoricalWordAnalysis/histwords-master'
SCRIPTS_PATH='vecanalysis'
export PYTHONPATH=${PYTHONPATH}:${HIST_HOME}:${HIST_HOME}/${SCRIPTS_PATH}

echo "Computing semantic displacement for words in $WORDS_FILE"
python3.8 semantic_displacement.py --start-year $START_YEAR --end-year $LAST_YEAR --year-inc $YEAR_INC --disp-year $START_YEAR --type SVD --dist $DIST $EMBS_DIR 4 $WORDS_FILE $OUTPUT_DIR

OUT_DIR=${OUTPUT_DIR}/$DIST/
sed 1d ${OUT_DIR}/displacement_info.txt | sort -k2n,2 > ${OUT_DIR}/displacement_sorted_year_2
sed 1d ${OUT_DIR}/displacement_info.txt | sort -k3n,3 > ${OUT_DIR}/displacement_sorted_year_3
