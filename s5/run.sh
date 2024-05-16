#!/usr/bin/env bash

#Modifying it for Santali data
#Modified by: Dipsankar Sinha, MCA, KIIT-DU, 15/05/2024

. ./cmd.sh
. ./path.sh

stage=5
#---------------------TO BE SET BY USER---------------------------------

# Do not include the last '/' in the directory path

path_to_data='/home/dipsankar/kaldi/egs/Santali_Kaldi/s5/Santali_data'


#-----------------------------------------------------------------------

nj=`cat /proc/cpuinfo | grep processor | wc -l`

. utils/parse_options.sh

set -euo pipefail

#-----------------------------------------------------------------------------------------------------
# Use this stage if you want to start from scrach, it will delete the existing files and start over

if [ $stage -le 1 ]; then
rm -rf data
rm -rf mfcc
rm -rf exp
fi
#-----------------------------------------------------------------------------------------------------
# Checks if the audio data folder is organized in the correct format
if [ $stage -le 0 ];then

  local/check_audio_data_folder.sh $path_to_data

fi
#------------------------------------------------------------------------------------------------------


train_dir=data/train
test_dir=data/test
lang_dir=data/lang


#---------------------------------------------------------------------------------------------
# Prepare Train data
if [ $stage -le 1 ];then
  local/train_data_prep.sh $path_to_data $train_dir
fi
#---------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------
# Prepare Test data
if [ $stage -le 2 ];then
  local/test_data_prep.sh $path_to_data $test_dir
fi
#---------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------
# Fix the train and test data directory to check if everything is sorted as per kaldi requirement
if [ $stage -le 3 ]; then

  utils/fix_data_dir.sh  $train_dir
  utils/fix_data_dir.sh  $test_dir
fi
#---------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------
# Compute features and validate train and test data directory
if [ $stage -le 4 ]; then
  mfccdir=mfcc
  
  for part in train test; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj data/$part exp/make_mfcc/$part $mfccdir
    utils/fix_data_dir.sh  data/$part
    steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
  done

  for part in train test; do
    utils/validate_data_dir.sh  data/$part
  done

fi
#---------------------------------------------------------------------------------------------


use_default_lexicon="no"
#---------------------------------------------------------------------------------------------
# Create Language model
if [ $stage -le 5 ]; then

  if [ $use_default_lexicon == "yes" ];then
    echo "Using default lexicon"
    mkdir -p data/local
    cp -r conf/dict data/local/

  elif [ $use_default_lexicon == "no" ];then
    echo "Using custom lexicon"
    [ ! -d data/local/dict ] && echo "Expected data/local/dict to exist with the following files: lexicon.txt, nonsilence_phones.txt, silence_phones.txt, optional_silence.txt" && exit 1;
    [ ! -f data/local/dict/lexicon.txt ] && echo "Expected data/local/dict/lexicon.txt to exist" && exit 1;
    [ ! -f data/local/dict/nonsilence_phones.txt ] && echo "Expected data/local/dict/nonsilence_phones.txt to exist" && exit 1;
    [ ! -f data/local/dict/silence_phones.txt ] && echo "Expected data/local/dict/silence_phones.txt to exist" && exit 1;
    [ ! -f data/local/dict/optional_silence.txt ] && echo "Expected data/local/dict/optional_silence.txt to exist" && exit 1;

  fi
  
  ./Create_LM.sh

fi
#---------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------
# Train a Monophone system
if [ $stage -le 6 ]; then
  
  steps/train_mono.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" $train_dir $lang_dir exp/mono
  
  (
    utils/mkgraph.sh $lang_dir exp/mono exp/mono/graph

    for test in test; do
      steps/decode.sh --nj $nj --cmd "$decode_cmd" exp/mono/graph $test_dir exp/mono/decode_$test
    done
  )&

  steps/align_si.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" $train_dir $lang_dir exp/mono exp/mono_ali_train
fi
echo "<--------------------------------------------------------------------------------------------------------------------------------------------------------------->"
echo 'Mono Phone Training is Completed and Word Error Rate is: '
cat exp/mono/decode_test/wer*|grep WER|sort|head -1
date
echo "<--------------------------------------------------------------------------------------------------------------------------------------------------------------->"
#---------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------
# Train a Delta + Delta-Delta triphone system
if [ $stage -le 7 ]; then
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2000 10000 $train_dir $lang_dir exp/mono_ali_train exp/tri1

  # decode using the tri1 model
  (
    utils/mkgraph.sh $lang_dir exp/tri1 exp/tri1/graph
    for test in test; do
      steps/decode.sh --nj $nj --cmd "$decode_cmd" exp/tri1/graph $test_dir exp/tri1/decode_$test
    done
  )&

  steps/align_si.sh --nj $nj --cmd "$train_cmd" $train_dir $lang_dir exp/tri1 exp/tri1_ali_train
fi
echo "<--------------------------------------------------------------------------------------------------------------------------------------------------------------->"
echo 'Tri Phone Training (tri1) is Completed and Word Error Rate is: '
cat exp/tri1/decode_test/wer*|grep WER|sort|head -1
date
echo "<--------------------------------------------------------------------------------------------------------------------------------------------------------------->"
#---------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------
# Train a LDA+MLLT system.
if [ $stage -le 8 ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd"  --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
    $train_dir $lang_dir exp/tri1_ali_train exp/tri2b

  # decode using the LDA+MLLT model
  (
    utils/mkgraph.sh $lang_dir exp/tri2b exp/tri2b/graph

    for test in test; do
      steps/decode.sh --nj $nj --cmd "$decode_cmd" exp/tri2b/graph $test_dir exp/tri2b/decode_$test
    done
  )&

  # Align utts using the tri2b model
  steps/align_si.sh  --nj $nj --cmd "$train_cmd" --use-graphs true $train_dir $lang_dir exp/tri2b exp/tri2b_ali_train
fi
#---------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------
# Train a tri3b model - LDA+MLLT+SAT
if [ $stage -le 9 ]; then
  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 $train_dir $lang_dir exp/tri2b_ali_train exp/tri3b

  # decode using the tri3b model
    utils/mkgraph.sh $lang_dir exp/tri3b exp/tri3b/graph

    for test in test; do
      steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" exp/tri3b/graph $test_dir exp/tri3b/decode_$test
        
    done
fi
echo "<--------------------------------------------------------------------------------------------------------------------------------------------------------------->"
echo 'Tri Phone Training (tri3b with LDA+MLLT+SAT) is Completed and Word Error Rate is: '
cat exp/tri3b/decode_test/wer*|grep WER|sort|head -1
date
echo "<--------------------------------------------------------------------------------------------------------------------------------------------------------------->"
exit 0
#---------------------------------------------------------------------------------------------
