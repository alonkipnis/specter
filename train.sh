export TRAIN_PATH="data/preprocessed/data-train.p"
export DEV_PATH="data/preprocessed/data-val.p"
export VOCAB_DIR="data/vocab/"
export NUM_EPOCHS="2"
export BATCH_SIZE="4"
export CUDA_DEVICE="-1" #change to "0" if GPU is availalbe
export CUDA_VISIBLE_DEVICES="0" 
export LAZY="true"
export TRAINING_DATA_INSTANCES="1768"
export BERT_REQUIRES_GRAD="all"
export BERT_MODEL="bert-pretrained"
export BERT_WEIGHTS="data/scibert_scivocab_uncased/scibert.tar.gz"
export BERT_VOCAB="data/scibert_scivocab_uncased/vocab.txt"
export INCLUDE_VENUE="false"
export MAX_SEQ_LEN="256"

python3 -m allennlp.run train experiment_configs/simple.jsonnet --include-package specter -s model-output/