#!/bin/sh


# Parse input params


while [ $# -gt 0 ]; do
  case $1 in
    -c|--corpus_dir)
      CORPUS="$2"
      shift # past argument
      shift # past value
      ;;
    -o|--output_dir)
      OUTPUT_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

HIST_HOME='/home/vivi/work/Projects/HistoricalWordAnalysis/histwords-master'
SCRIPTS_PATH='sgns/hyperwords'
export PYTHONPATH=${PYTHONPATH}:${HIST_HOME}:${HIST_HOME}/${SCRIPTS_PATH}

min_count=3
CORPUS2PAIRS_OPTS='--thr '$min_count   ## frequency threshold for cooccurrences
PMI2SVD_OPTS='--dim 100'      ## dimensionality of vectors
SVD2TEXT_OPTS=''


# Create collection of word-context pairs
mkdir -p $OUTPUT_DIR

## process each time slice
for time_slice in $CORPUS/* 
do 

   time=`basename $time_slice `
   pairs=${OUTPUT_DIR}/${time}.pairs
   counts=${OUTPUT_DIR}/${time}.counts
   pmi=${OUTPUT_DIR}/${time}.pmi
   svd=${OUTPUT_DIR}/${time}.svd
   vectors=${OUTPUT_DIR}/${time}.vectors
 
   echo "Processing time slice $time ..."
   
   echo "   extracting cooccurrences and counts "
   python3.8 ${SCRIPTS_PATH}/hyperwords/corpus2pairs.py $CORPUS2PAIRS_OPTS $time_slice > $pairs
   ${SCRIPTS_PATH}/scripts/pairs2counts.sh $pairs > $counts	
   python3.8 ${SCRIPTS_PATH}/hyperwords/counts2vocab.py $counts


   echo "   calculating PMI matrices "
   # Calculate PMI matrices for each collection of pairs
   python3.8 ${SCRIPTS_PATH}/hyperwords/counts2pmi.py $COUNTS2PMI_OPTS $counts $pmi


   echo "   creating SVD embeddings "
   # Create embeddings with SVD
   python3.8 ${SCRIPTS_PATH}/hyperwords/pmi2svd.py $PMI2SVD_OPTS $pmi $svd
   #cp $OUTPUT_DIR/${time}.pmi.words.vocab $OUTPUT_DIR/${time}.svd.words.vocab
   #cp $OUTPUT_DIR/${time}.pmi.contexts.vocab $OUTPUT_DIR/${time}.svd.contexts.vocab


   echo "   saving embeddings in textual format "
   # Save the embeddings in the textual format 
   python3.8 ${SCRIPTS_PATH}/hyperwords/svd2text.py $SVD2TEXT_OPTS $svd $vectors

done


#python3.8 vecanalysis/seq_procrustes.py --start-year 1150 --end-year 2050 --year-inc 100 --min-count 3 ../Data/time_sliced/embeddings/text/ SVD ../Data/time_sliced/embeddings/text/
python3.8 vecanalysis/seq_procrustes.py --start-year 1 --end-year 3 --year-inc 1 --min-count $min_count $OUTPUT_DIR SVD $OUTPUT_DIR

# Remove temporary files
#rm $CORPUS.clean
#rm $OUTPUT_DIR/*pairs
#rm $OUTPUT_DIR/*counts*
#rm $OUTPUT_DIR/*pmi*
#rm $OUTPUT_DIR/*svd*
