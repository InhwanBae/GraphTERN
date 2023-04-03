#!/bin/bash
echo "Start evaluation task queues"

# Hyperparameters
name_array=("eth" "hotel" "univ" "zara1" "zara2")
device_id_array=(0 1 2 3 4)
prefix="graph-tern_"
suffix="_experiment"

# Arguments
while getopts p:s:d:i: flag
do
  case "${flag}" in
    p) prefix=${OPTARG};;
    s) suffix=${OPTARG};;
    d) dataset_array=(${OPTARG});;
    i) device_id_array=(${OPTARG});;
    *) echo "usage: $0 [-p PREFIX] [-s SUFFIX] [-d \"eth hotel univ zara1 zara2\"] [-i \"0 1 2 3 4\"]" >&2
      exit 1 ;;
  esac
done

if [ ${#dataset_array[@]} -ne ${#device_id_array[@]} ]
then
    printf "Arrays must all be same length. "
    printf "len(dataset_array)=${#dataset_array[@]} and len(device_id_array)=${#device_id_array[@]}\n"
    exit 1
fi

# Signal handler
PID_array=()

sighdl ()
{
  echo "Kill evaluation processes"
  for (( i=0; i<${#name_array[@]}; i++ ))
  do
    kill ${PID_array[$i]}
  done
  echo "Done."
  exit 0
}

trap sighdl SIGINT SIGTERM

# Start testing tasks
for (( i=0; i<${#name_array[@]}; i++ ))
do
  printf "Testing ${name_array[$i]}"
  CUDA_VISIBLE_DEVICES=${device_id_array[$i]} python3 test.py \
  --tag "${prefix}""${name_array[$i]}""${suffix}" &
  PID_array[$i]=$!
  printf " job ${#PID_array[@]} pid ${PID_array[$i]}\n"
  wait ${PID_array[$i]}
done

echo "Done."
