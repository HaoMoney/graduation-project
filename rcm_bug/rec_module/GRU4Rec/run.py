#-*- coding: utf-8 -*-

import sys

import numpy as np
import pandas as pd
import gru4rec
import evaluation
import matplotlib.pyplot as plt

PATH_TO_TRAIN = '../report_labels_train'
PATH_TO_TEST = '../report_labels_test'

if __name__ == '__main__':
    data = pd.read_csv(PATH_TO_TRAIN, sep='\t')
    valid = pd.read_csv(PATH_TO_TEST, sep='\t')
    gru = gru4rec.GRU4Rec(n_epochs=10,loss='top1', final_act='tanh', hidden_act='relu', layers=[512], batch_size=32, dropout_p_hidden=0.5, learning_rate=0.01, momentum=0.0, time_sort=False)
    gru.fit(data)
    res_gru = evaluation.evaluate_sessions_batch(gru, valid, None,cut_off=21)
    acc=res_gru[0]
    mrr=res_gru[1]
    print "accuracy@20:%f,mrr:%f" % (acc,mrr)
