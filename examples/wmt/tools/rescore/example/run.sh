working_dir=./work
bin_dir=../bin

# step 1: train the rescoring weights with kbmira
python ../train.py --nbest nbest --ref ref --working-dir ${working_dir} --bin-dir ${bin_dir}

# step 2: rewrite the score based on new feature weights
python ../rescore.py ${working_dir}/rescore.ini < nbest > nbest.rescored

# step 3: get the best candidate based on new score
python ../topbest.py < nbest.rescored > topbest.output
