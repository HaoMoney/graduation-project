#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def evaluate_sessions_batch(pr, test_data, items=None, cut_off=2, batch_size=16, break_ties=False, session_key='SessionId', item_key='ItemId', time_key='Time'):
    '''
    Evaluates the GRU4Rec network wrt. recommendation accuracy measured by accuracy@N and MRR.

    Parameters
    --------
    pr : gru4rec.GRU4Rec
        A trained instance of the GRU4Rec network.
    test_data : pandas.DataFrame
        Test data. It contains the transactions of the test set.It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to. If None, all items of the training set are used. Default value is None.
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list; N for accuracy@N and MRR). Defauld value is 20.
    batch_size : int
        Number of events bundled into a batch during evaluation. Speeds up evaluation. If it is set high, the memory consumption increases. Default value is 100.
    break_ties : boolean
        Whether to add a small random number to each prediction value in order to break up possible ties, which can mess up the evaluation. 
        Defaults to False, because (1) GRU4Rec usually does not produce ties, except when the output saturates; (2) it slows down the evaluation.
        Set to True is you expect lots of ties.
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')
    
    Returns
    --------
    out : tuple
        (accuracy@N, MRR)
    
    '''
    test_data.sort_values([session_key, time_key], inplace=True)
    offset_sessions = np.zeros(test_data[session_key].nunique()+1,dtype=np.int32)
    offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()
    evalutation_point_count = 0
    mrr, acc = 0.0, 0.0
    if len(offset_sessions) - 1 < batch_size:
        batch_size = len(offset_sessions) - 1
    iters = np.arange(batch_size).astype(np.int32)
    maxiter = iters.max()
    start = offset_sessions[iters]
    end = offset_sessions[iters+1]
    in_idx = np.zeros(batch_size,dtype=object)
    sampled_items = (items is not None)
    while True:
        valid_mask = iters >= 0
        if valid_mask.sum() == 0:
            break
        start_valid = start[valid_mask]
        minlen = (end[valid_mask]-start_valid).min()
        in_idx[valid_mask] = test_data.ItemId.values[start_valid]
        for i in range(minlen-1):
            out_idx = test_data.ItemId.values[start_valid+i+1]
            if sampled_items:
                uniq_out = np.unique(np.array(out_idx, dtype=np.int32))
                preds = pr.predict_next_batch(iters, in_idx, np.hstack([items, uniq_out[~np.in1d(uniq_out,items)]]), batch_size)
            else:
                preds = pr.predict_next_batch(iters, in_idx, None, batch_size) #TODO: Handling sampling?
            preds.fillna(0, inplace=True)
            if break_ties:
                preds += np.random.rand(*preds.values.shape) * 1e-8
            in_idx[valid_mask] = out_idx
            if sampled_items:
                others = preds.ix[items].values.T[valid_mask].T
                targets = np.diag(preds.ix[in_idx].values)[valid_mask]
                ranks = (others > targets).sum(axis=0) 
            else:
                ranks = (preds.values.T[valid_mask].T > np.diag(preds.ix[in_idx].values)[valid_mask]).sum(axis=0) + 1
            rank_ok = ranks < cut_off
            acc += rank_ok.sum()
            mrr += (1.0 / ranks[rank_ok]).sum()
            evalutation_point_count += len(ranks)
        start = start+minlen-1
        mask = np.arange(len(iters))[(valid_mask) & (end-start<=1)]
        for idx in mask:
            maxiter += 1
            if maxiter >= len(offset_sessions)-1:
                iters[idx] = -1
            else:
                iters[idx] = maxiter
                start[idx] = offset_sessions[maxiter]
                end[idx] = offset_sessions[maxiter+1]
    return acc/evalutation_point_count, mrr/evalutation_point_count
