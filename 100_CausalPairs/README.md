Basic outline of workflow: 

preprocess for task 1 --> run task 1 --> preprocess for task 2 --> run task 2 --> preprocess for task 3 --> run task 3 --> preprocess for phraseBERT



To preprocess data for task 1: `python preprocess --task 1 --data_path RAW_DATA_PATH`, 

​	use `--num_sent NUM_SENT` to indicate number of sentences in each chunk, if not 1. 

To run task 1: 

`python run_seqbase.py --seq_val_file TASK_1_DATA_PATH --do_predict --model_name_or_path tanfiona/unicausal-seq-baseline --output_dir OUT_DIR `



To preprocess for task 2: `python preprocess --task 2 --data_path TASK_1_DATA_PATH --preds_path TASK_1_OUTPUT_PATH`

To run task 2: 

`python run_seqbase.py --seq_val_file TASK_2_DATA_PATH --do_predict --model_name_or_path tanfiona/unicausal-tok-baseline --output_dir OUT_DIR`



To preprocess for task 3: `python preprocess --task 3 --data_path TASK_2_DATA_PATH --preds_path TASK_2_OUTPUT_PATH`

To run task 3:

`python run_seqbase.py --seq_val_file TASK_3_DATA_PATH --do_predict --model_name_or_path tanfiona/unicausal-pair-baseline --output_dir OUT_DIR` 



To preprocess for PhraseBERT:

​	If only want sentences with pairs added and not want to include args: `python get_res --raw_path RAW_DATA_PATH --task3out_path TASK_3_OUTPUT --task3in_path TASK_3_DATA_PATH`

​	If want to include args: `python get_res --raw_path RAW_DATA_PATH --task3out_path TASK_3_OUTPUT --task2in_path TASK_3_DATA_PATH --task2out_path TASK_2_OUTPUT_PATH --args`



For any preprocessing step, using command `--out_path OUT_PATH` can change from default save path.