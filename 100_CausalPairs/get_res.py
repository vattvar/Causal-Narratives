import numpy as np
import pandas as pd

from preprocess_utils import *

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process result data from causal mining tasks in order to feed into PhraseBERT topic modeling. Currently specifically for POLUSA dataset."
    )
    parser.add_argument(
        "--raw_path", type=str, default=None, help= "A csv file containing raw data."
    )
    parser.add_argument(
        "--task3out_path", type=str, default=None, help= "A csv file containing raw data."
    )    
    parser.add_argument(
        "--task2in_path", type=str, default="task2_in.csv", help= "A csv file containing preprocessed data for task2."
    )
    parser.add_argument(
        "--task2out_path", type=str, default="task2_out.txt", help= "A txt file containing task 2 (span detection) predtions."
    )
    parser.add_argument(
        "--task3in_path", type=str, default="task3_in.csv", help= "A csv file containing preprocessed data for task 3."
    )
    parser.add_argument(
        "--args", action='store_true', help= "Indicator of including args in final dataframe."
    )
    parser.add_argument(
        "--out_path", type=str, default="result.csv", help= "Path to save the results."
    )    

    args = parser.parse_args()


    return args



def preprocess_task3_args(file_path, file_path_tags):
    """ 
    Preprocess for task 3, but with args included in output
    """
    
    df = pd.read_csv(file_path)
    df_span = pd.read_csv(file_path_tags, delimiter='\t')

    df_new = pd.merge(df, df_span, on=['index'], how='inner')
    print("data_loaded")
    
    df_new['pred_list'] = df_new.pred.apply(lambda x: x[1:-1].split(", "))
    df_new['c_e_idx'] = df_new.pred_list.apply(lambda x: tag2idx(x))

    df_new['text_w_pairs'] = df_new.apply(lambda x: tag_arg(x['text'], x['c_e_idx']), axis=1)
    df_new['args'] = df_new.apply(lambda x: get_args(x['text'], x['c_e_idx']), axis=1)
    df_new['eg_id'] = df_new.text_w_pairs.apply(lambda x: list(range(len(x))))
    #df_new = df_new[['corpus', 'doc_id', 'sent_id', 'eg_id', 'index', 'text', 'text_w_pairs',
    #   'seq_label', 'pair_label', 'context', 'num_sents']]
                            
    df_new = df_new.explode(['eg_id', 'text_w_pairs', 'args'], ignore_index=True)
    df_new['index'] = df_new['corpus'].astype(str) + "_" + df_new['doc_id'].astype(str) + "_" + df_new['sent_id'].astype(str) + "_" + df_new['eg_id'].astype(str)
                                                

    return df_new[df_new['eg_id'].notna()]



def main():
    args = parse_args()
    assert args.raw_path != None

    raw = pd.read_csv(args.raw_path)
    pred = pd.read_csv(args.task3out_path, delimiter='\t')
    if args.args: 
        task3in = preprocess_task3_args(args.task2in_path, args.task2out_path)
    else:
        task3in = pd.read_csv(args.task3in_path)
    task3in_w_pred = pd.merge(task3in, pred, on='index')
    causals = task3in_w_pred[task3in_w_pred.pair_pred == 1]
    if args.args:
        res = causals[['doc_id', 'index', 'text', 'args']]
        merged = pd.merge(raw, res.rename(columns={'doc_id':'id'}), on='id')[['id', 'index', 'outlet', 'political_leaning', 'date_publish', 'text', 'args']]
    else:
        res = causals[['doc_id', 'index', 'text', 'text_w_pairs']]
        merged = pd.merge(raw, res.rename(columns={'doc_id':'id'}), on='id')[['id', 'index', 'outlet', 'political_leaning', 'date_publish', 'text', 'text_w_pairs']]


    merged.to_csv(args.out_path, index=False)







if __name__ == "__main__":
    main()