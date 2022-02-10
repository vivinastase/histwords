#!/bin/sh

pairs_file=$1
sort -T . $pairs_file | uniq -c
