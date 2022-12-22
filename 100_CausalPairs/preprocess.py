import pandas as pd
import numpy as np
import nltk.data
nltk.download('punkt')

from preprocess_utils import *

from tqdm import tqdm
import argparse
import os
import warnings

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess data for the three causal mining sub-tasks. Currently specifically for POLUSA dataset."
    )
    parser.add_argument(
        "--task", type=int, required=True, default=None, help= "A number indicating which task is preprocessing for."
    )
    parser.add_argument(
        "--data_path", type=str, default=None, help= "A csv file containing the raw data (if doing task 1) or preprocessed data for previous task (if doing task 2 or task 3)."
    )
    parser.add_argument(
        "--preds_path", type=str, default=None, help= "A txt file containing the sequence classification predictions."
    )
    parser.add_argument(
        "--num_sent", type=int, default=0, help="A number indicating how many sentences included in each chunk. Default is 1."
    )
    parser.add_argument(
        "--out_path", type=str, default=None, help="A save path ending in .csv for preprocessed data. Default is task#_in.csv, where # is task number"
    )
    args = parser.parse_args()

    if args.out_path == None:
        args.out_path = f"task{args.task}_in.csv"

    # Sanity checks

    if args.task == 1:
        if args.data_path == None:
            raise ValueError("Need a valid data file containing the original data")
        if args.num_sent == 0:
            warnings.warn("No valid number of sentence is given, so defaulting to 1.")
            args.num_sent = 1
    elif args.task == 2:
        if args.data_path == None:
            raise ValueError("Need a valid data file containing the preprocessed data for task1 using command --data_path")
        if args.preds_path == None:
            raise ValueError("Need a valid txt file containing the predictions from task1(sequence classification), using the command --preds_path")
    elif args.task == 3:
        if args.data_path == None:
            raise ValueError("Need a valid data file containing the preprocessed data for task2 using command --data_path")
        if args.preds_path == None:
            raise ValueError("Need a valid txt file containing the predictions from task2(span detection), using the command --preds_path")

        
    return args


def preprocess_task1(file_path, n):
    """ Preprocess raw data for task 1.

        file_path: path to the file that contains polusa data, containing 'body' and 'id' column
        n: the number of sentences to be included in each example/chunk. 1-indexed
        
        returns a dataframe usable in run_task1.py
    """
    # read data into dataframe
    df = pd.read_csv(file_path)

    # returns None if n is a bad number
    if n<1 or type(n) != int:
        return None
    
    # helper function to tokenize article into sentences, and join n number of sentences as one example.
    def get_text(body, n):
        tokenized = tokenizer.tokenize(body)
        splitted = []
        for i in range(len(tokenized)):
            splitted.append(" ".join(tokenized[i:i+n]))
        
        return splitted
    
    # tokenize data
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    df['text'] = df['body'].copy()
    for i in tqdm(range(len(df.body))):
        df.text.iloc[i] = get_text(df.body.iloc[i], n)
    
    df = df.explode('text', ignore_index=True)  
    
    # clean data by removing \n symbols and removing sentences longer than BERT embeddings(512)
    df.text = df.text.apply(lambda x: x.replace("\n", " "))
    df = df.loc[df.text.str.len()<=512]
    
    # convert polusa index to unicausal index
    df['corpus'] = "polusa"
    df['doc_id'] = df['id']
  
    df['sent_id'] = df.groupby('id').cumcount()
    if n>1:
        df['sent_id'] = df['sent_id'].apply(lambda x: ";".join([str(y) for y in list(range(x, x+n))]))
    df['eg_id'] = 0
    df['index'] = "polusa_" + df['doc_id'].astype(str) + "_" + df['sent_id'].astype(str) + "_" + df['eg_id'].astype(str)
    
            
    # add other dummy columns not related to task1 but required by modeling code
    df['text_w_pairs'] = ' '
    df['seq_label'] = 1
    df['pair_label'] = 1
    df['context'] = ''
    df['num_sents'] = n
    
    
    return df[['corpus', 'doc_id', 'sent_id', 'eg_id', 'index', 'text', 'text_w_pairs',
       'seq_label', 'pair_label', 'context', 'num_sents']]





def preprocess_task2(file_path_task1_in, file_path_preds):
    """ Preprocess data and task 1 output for task 2.
        
        file_path: path to the file that contains the original dataframe used for task1 (task1_in)
        file_path_preds: path to the file contains dataframe containing predictions out from task1 (task1_out)
        
        task1_out contains only sequences predicted to be causal in task 1.
        
        returns a dataframe usable in task2 toke_base.py
    """
    
    # read in original preprocessed data for task1 and sequence classification predictions and 
    df = pd.read_csv(file_path_task1_in) 
    df_preds = pd.read_csv(file_path_preds, delimiter="\t")

    # merge data
    causals = pd.merge(df, df_preds[df_preds["seq_pred"]==1], on=['index', 'seq_label'], how='inner')
    causals = causals.drop(columns='seq_pred')
    causals['text_w_pairs'] = causals['text']

    return causals





def preprocess_task3(file_path, file_path_tags):
    """  Preprocess data and task 2 output for task 3.
    
        file_path: path to the file that contains the original dataframe used for task1 (task1_in)
         file_path_tags: path to the file contains dataframe containing token tags from task2 (task2_out)
    """
    
    df = pd.read_csv(file_path)
    df_span = pd.read_csv(file_path_tags, delimiter='\t')

    df_new = pd.merge(df, df_span, on=['index'], how='inner')
    #print("data_loaded")
    
    
    df_new['pred_list'] = df_new.pred.apply(lambda x: x[1:-1].split(", "))
    df_new['c_e_idx'] = df_new.pred_list.apply(lambda x: tag2idx(x)) # get lists of cause effect in the form of begin, end indices

    df_new['text_w_pairs'] = df_new.apply(lambda x: tag_arg(x['text'], x['c_e_idx']), axis=1) # add <ARG><\ARG> labels in text. combinations are stored in a list for each text
    df_new['eg_id'] = df_new.text_w_pairs.apply(lambda x: list(range(len(x)))) # a separate example id for each combination of cause and effect, stored in a list for each text
    df_new = df_new[['corpus', 'doc_id', 'sent_id', 'eg_id', 'index', 'text', 'text_w_pairs',
       'seq_label', 'pair_label', 'context', 'num_sents']]
                            
    df_new = df_new.explode(['eg_id', 'text_w_pairs'], ignore_index=True) # explode lists to make each one a separate example, and match with example id.
    df_new['index'] = df_new['corpus'].astype(str) + "_" + df_new['doc_id'].astype(str) + "_" + df_new['sent_id'].astype(str) + "_" + df_new['eg_id'].astype(str) # update index
                                                

    return df_new[df_new['eg_id'].notna()]




def main():
    args = parse_args()
    assert args.data_path != None

    if args.task == 1:
        print("preprocessing for task 1...")
        task1_in = preprocess_task1(args.data_path, args.num_sent)
        print("Saving preprocessed data...")
        task1_in.to_csv(args.out_path, index=False)
        print("Done!")
    elif args.task == 2:
        task1_in = args.data_path
        task1_pred = args.preds_path
        print("preprocessing for task 2...")
        task2_in = preprocess_task2(task1_in, task1_pred)
        print("Saving preprocessed data...")
        task2_in.to_csv(args.out_path, index=False)
        print("Done!")
    elif args.task == 3:
        task2_in = args.data_path
        task2_pred = args.preds_path
        print("preprocessing for task 3...")
        task3_in = preprocess_task3(task2_in, task2_pred)
        print("Saving preprocessed data...")
        task3_in.to_csv(args.out_path, index=False)
        print("Done!")








if __name__ == "__main__":
    main()


