from itertools import product



def tag2idx(tags):
    """ Convert raw BIO-CE token tags to (begin index, end index) tuples for causes and effects.

        tags: a list of raw BIO-CE tags for a single sequence.

        returns two lists of tuples, first is for causes and second is for effects, containing all detected causes and effects in a single sequence.
        Each tuple in the lists is a pair of indices indicating the beginning and end index of either cause or effect.
    """
    #lists to store index of causes and effect spans in a sequence
    cause_idx = []
    effect_idx = []
    tags.append("end")
    current_idx = 0
    current_tag = tags[current_idx]
    while current_tag != "end":
        if current_tag in ["'B-C'", "'I-C'"]:
            current_cause_begin = current_idx
            current_idx += 1
            current_tag = tags[current_idx]
            
            while current_tag == "'I-C'":
                current_idx +=1
                current_tag = tags[current_idx]
            current_cause_end = current_idx-1
            cause_idx.append((current_cause_begin, current_cause_end))
            
        elif current_tag in ["'B-E'", "'I-E'"]:

            current_effect_begin = current_idx
            current_idx += 1
            current_tag = tags[current_idx]
            
            while current_tag == "'I-E'":
                current_idx +=1
                current_tag = tags[current_idx]
            current_effect_end = current_idx-1
            effect_idx.append((current_effect_begin, current_effect_end))
        
        else:
            current_idx +=1
            current_tag = tags[current_idx]
        
    return cause_idx, effect_idx





def tag_arg(seq, tag_idx):
    """ Add argument tags i.e. <ARG></ARG> to text.
        
        seq is a string containing the orginal text without pairs.
        tag_idx is a tuple containing two lists of tuples.
        
        return a string conataining argtags"""
    
    c_list = tag_idx[0]
    e_list = tag_idx[1]
    seq_list = seq.split()
    text_w_pair_list = [] # master list

    pairs = list(product(c_list,e_list))
    for pair in pairs:
        seq_w_pair_list = [] # inner list for each pair
        c_idx_begin = pair[0][0]
        c_idx_end = pair[0][1]
        e_idx_begin = pair[1][0]
        e_idx_end = pair[1][1]
        for i in range(len(seq_list)):
            term = seq_list[i]
            if i == c_idx_begin:
                term = "<ARG0>" + term
            if i == c_idx_end:
                term = term + "</ARG0>"
            if i == e_idx_begin:
                term = "<ARG1>" + term
            if i == e_idx_end:
                term = term + "</ARG1>" 
            seq_w_pair_list.append(term)
        text_w_pair_list.append(" ".join(seq_w_pair_list))
    
    return text_w_pair_list



def get_args(seq, tag_idx):
    """ Extract the cause and effect arguments from sequence.

        seq is a string containing the orginal text without pairs.
        tag_idx is a tuple containing two lists of tuples.
        
        Return a list of argument string pairs. The list contains all possible combinations of causes and effects.
    """
    
    c_list = tag_idx[0]
    e_list = tag_idx[1]
    seq_list = seq.split()
    arg_pair_list = [] # master list

    pairs = list(product(c_list,e_list))
    for pair in pairs:
        args_pair = [] # inner list for each pair
        c_idx_begin = pair[0][0]
        c_idx_end = pair[0][1]
        e_idx_begin = pair[1][0]
        e_idx_end = pair[1][1]
        args_pair.append(seq[c_idx_begin:c_idx_end+1])
        args_pair.append(seq[e_idx_begin:e_idx_end+1])
        
        arg_pair_list.append(args_pair)
    
    return arg_pair_list