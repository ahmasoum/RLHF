#!/usr/bin/env bash
set -e

mkdir -p data
cd data

wget https://zenodo.org/records/5068253/files/twitter.jsonl

echo "Download completed: data/twitter.jsonl"
