RAW_DATA_DIR = $(shell pwd)/twitter_br_lms/data/raw
INTERIM_DATA_DIR = $(shell pwd)/twitter_br_lms/data/interim
PROCESSED_DATA_DIR = $(shell pwd)/twitter_br_lms/data/processed
MODELS_DIR = $(shell pwd)/twitter_br_lms/models
CACHE_DIR = ~/.cache/twitter-br

init:
	mkdir -p $(CACHE_DIR)
	wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -O $(CACHE_DIR)/lid.176.bin
	pip install -r requirements.txt
	pip install -e .

download_all: download_ptsa download_ttsbr download_j2015_2016 download_covid19_4m download_brasnam2018 download_def download_mariocovid19
	echo "Downloading all data"

download_mariocovid19:
	# Download MarioCOVID19
	cd $(RAW_DATA_DIR) && gshell download --with-id 1WKUZGMurOlWSm4k-oO40xAWL2l2l0EC8 --recursive

download_ptsa:
	# Download PortugueseTweetsforSentimentAnalysis
	cd $(RAW_DATA_DIR) && gshell download --with-id 1MoV2LpJjWroBt7sHoujFPMS-ap8pcNGs --recursive

download_ttsbr:
	# Download TweetSentsBR
	cd $(RAW_DATA_DIR) && gshell download --with-id 1H1v3H2a_rhJ9Mg4bJ3wz4WsfPbnS5iXK --recursive

download_j2015_2016:
	# Download Joao2015-20016
	cd $(RAW_DATA_DIR) && gshell download --with-id 1H1v3H2a_rhJ9Mg4bJ3wz4WsfPbnS5iXK --recursive

download_covid19_4m:
	# Download COVID19-4M
	cd $(RAW_DATA_DIR) && gshell download --with-id 1FvTXR7uPo0jtkBQ01B8ED5fYNuS0skkM --recursive

download_brasnam2018:
	# Download BraSNAM2018
	cd $(RAW_DATA_DIR) && gshell download --with-id 1OD_Xr8ijs6HJyQGUMB9Nws4me3E-4hOj --recursive

download_def:
	# Download Tweets sobre deficiencia
	cd $(RAW_DATA_DIR) && gshell download --with-id 1eHzQMx1_TFJVkUW318_GQfoav-Urs4k6 --recursive

split_data:
	# Create train/val splits
	python3 -m twitter_br_lms.split_data \
		--data_path $(INTERIM_DATA_DIR) \
		--output_path $(PROCESSED_DATA_DIR) \
		--drop_duplicates

filter-pt-br:
	# Filter pt-br tweets from raw datasets
	python3 -m twitter_br_lms.filter_lang \
		--data_path $(RAW_DATA_DIR) \
		--cache_dir $(CACHE_DIR) \
		--model_name "lid.176.bin" \
		--output_path $(INTERIM_DATA_DIR)

train-tokenizer:
	python3 -m twitter_br_lms.train_tokenizer \
		--train_file $(PROCESSED_DATA_DIR)/train.csv \
		--output_path $(MODELS_DIR)/$(tokenizer) \
		--tokenizer $(tokenizer) \
		--vocab_size 50200 \
		--uncased

roberta-train-check:
	python3 -m twitter_br_lms.mlm \
		--output_dir $(MODELS_DIR)/roberta \
		--train_file $(PROCESSED_DATA_DIR)/train.csv \
		--validation_file $(PROCESSED_DATA_DIR)/val.csv \
		--model_name_or_path roberta-base \
		--tokenizer_name $(MODELS_DIR)/roberta-tokenizer \
		--preprocessing_num_workers 8 \
		--do_train \
		--seed 42 \
		--overwrite_output_dir \
		--per_device_train_batch_size 16 \
		--save_total_limit 3 \
		--eval_accumulation_steps 100 \
		--fp16 \
		--max_train_samples 100 \
		--max_eval_samples 100 \
		--num_train_epochs 1 \
		--max_seq_length 512 \
		--debugging

roberta-train:
	python3 -m twitter_br_lms.mlm \
		--output_dir $(MODELS_DIR)/roberta \
		--train_file $(PROCESSED_DATA_DIR)/train.csv \
		--validation_file $(PROCESSED_DATA_DIR)/val.csv \
		--model_name_or_path roberta-base \
		--tokenizer_name $(MODELS_DIR)/roberta-tokenizer \
		--preprocessing_num_workers 8 \
		--do_train \
		--seed 42 \
		--overwrite_output_dir \
		--per_device_train_batch_size 16 \
		--save_total_limit 3 \
		--eval_accumulation_steps 100 \
		--max_seq_length 512 \
		--evaluation_strategy "steps" \
		--eval_steps 250000 \
		--max_train_samples 40000000 \
		--max_eval_samples 400000 \
		--fp16

roberta-eval:
	python3 -m twitter_br_lms.mlm \
		--model_name_or_path $(MODELS_DIR)/twitter-br \
		--train_file $(PROCESSED_DATA_DIR)/train.csv \
		--validation_file $(PROCESSED_DATA_DIR)/val.csv \
		--tokenizer_name $(MODELS_DIR)/roberta-tokenizer \
		--preprocessing_num_workers 8 \
		--do_eval \
		--seed 42 \
		--per_device_train_batch_size 16 \
		--save_total_limit 3 \
		--eval_accumulation_steps 100 \
		--fp16 \
		--output_dir $(MODELS_DIR)/
