Testing:
bash hw2_seq2seq.sh [testing data dir] [output filename]
eg. bash hw2_seq2seq.sh ../mlds_hw2_1_data/testing_data/ ../output.txt

Training:
python model_seq2seq.py [training data dir] [training labels filename] [epoch #] [model name]
eg. python model_seq2seq.py ../mlds_hw2_1_data/training_data/
		../mlds_hw2_1_data/training_label.json 10 ../model_tmp

