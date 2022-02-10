#!/bin/sh

./corpus2svd_full_proc.sh --corpus_dir ../Data/time_sliced/text/ --output_dir ../Data/time_sliced/embeddings/text
./corpus2svd_full_proc.sh --corpus_dir ../Data/time_sliced/lemmatized/ --output_dir ../Data/time_sliced/embeddings/lemmatized

