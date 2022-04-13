#!/bin/sh

CORPUS_DIR=/home/vivi/work/Projects/HistoricalWordAnalysis/Data/child_speech_corpus
PREFIX=child_
OUT_DIR=${CORPUS_DIR}/embeddings

while [ $# -gt 0 ]; do
  case $1 in
    -c|--corpus)
      NAME="$2"
      shift # past argument
      shift # past value
      ;;
    -p|--prefix)
      DAYS="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done



for x in 'text' 'lemmatized'
do
  ./corpus2svd_full_proc.sh --corpus_dir ${CORPUS_DIR}/${PREFIX}$x/ --output_dir ${OUT_DIR}/${PREFIX}$x/

done 
