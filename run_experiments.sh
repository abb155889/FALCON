#!/usr/bin/env bash

# Shell script to run the anomaly detection pipeline
# Usage: ./run_experiments.sh [-d dataset] [-s subdataset] [-o output_dir] \
#                           [-a mvtec_ad_path] [-c mpdd_path] [-e visa_path] \
#                           [--epochs epochs] [--data_limit limit]

# Default arguments
DATASET="visa" # choices=['mvtec_ad', 'mpdd', 'visa'])
SUBDATASET="all" # sub class
OUTPUT_DIR="./experiment_results" # output dir
MVTEC_AD_PATH="./MVTecAD" # mvtec_ad dataset path 
MPDD_PATH="./MPDD" # mpdd dataset path 
VISA_PATH="./ViSA" # visa datasaet path 
EPOCHS=100 # epochs
DATA_LIMIT=5 # few shot setting choices=[2, 5, 10]

# Parse command-line options
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -d|--dataset)
      DATASET="$2"; shift; shift;;
    -s|--subdataset)
      SUBDATASET="$2"; shift; shift;;
    -o|--output_dir)
      OUTPUT_DIR="$2"; shift; shift;;
    -a|--mvtec_ad_path)
      MVTEC_AD_PATH="$2"; shift; shift;;
    -c|--mpdd_path)
      MPDD_PATH="$2"; shift; shift;;
    -e|--visa_path)
      VISA_PATH="$2"; shift; shift;;
    --epochs)
      EPOCHS="$2"; shift; shift;;
    --data_limit)
      DATA_LIMIT="$2"; shift; shift;;
    *)
      echo "Unknown option: $1"; exit 1;;
  esac
done

# Ensure output directory exists
mkdir -p "${OUTPUT_DIR}"

# Run the Python script (replace `script.py` with your filename)
python falcon.py \
  -d "${DATASET}" \
  -s "${SUBDATASET}" \
  -o "${OUTPUT_DIR}" \
  -a "${MVTEC_AD_PATH}" \
  -c "${MPDD_PATH}" \
  -e "${VISA_PATH}" \
  --epochs "${EPOCHS}" \
  --data_limit "${DATA_LIMIT}"

