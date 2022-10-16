import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal


def vote_count(y, y_val, preds_nonUS):
    # IMPORTANT: DO YOU NEED THE RANDOM SEQUENCE, OR CAN YOU JUST VOTE WITH YES/NO?
    print(type(y_val))
    print(y_val.shape)
    print(y_val[preds_nonUS == 1][:10])
    print(y_val[y == 1][:3])
    tmp = y_val[y == 1]
    print(tmp.shape)

    preds_set_hit = np.zeros(y.shape[0] // 9)
    for i in range(9):
        # rows = hit_seq[i::12]
        rows = preds_nonUS[i::9]
        # set_hit = np.vstack((set_hit, rows))
        preds_set_hit = np.vstack((preds_set_hit, rows))
    print(np.shape(preds_set_hit))
    plt.hist(sum(preds_set_hit[1:, :]))  # TELLS HOW MANY HITS PER 9 FLASHES, SHOULD HAVE 1 ONLY
    preds_set_hit = np.transpose(preds_set_hit[1:, :])
    print(np.shape(preds_set_hit))
    print(preds_set_hit[0:10])


def print_predict_and_truth(y, y_val, preds_nonUS, num_markers):
    vote_result = []
    for ith in range(y.shape[0] // 45):
        votes = y_val[ith * 45: (ith + 1) * 45][preds_nonUS[ith * 45: (ith + 1) * 45] == 1]
        modes = pd.DataFrame(votes).value_counts().index.tolist()
        vote_result += [modes[0][0]]
    print("vote_result", vote_result)
    print("vote_truth ", num_markers)
    print("correct percentage:", (sum(np.array(num_markers) == np.array(vote_result)) / 27))

    # New one
    # for wth in range(10):
    #     # CHANGE HERE
    #     # MOVING AVERAGE / TRIANGLE FILTER
    #     # MA3pt_filter = [1/(2**(wth*2)), 1, 1/(2**wth)]
    #     # MA3pt_filter = [0.1,1,0.1+wth/10] # [0.1, 1, 0.2-0.3]
    wth = 1
    MA3pt_filter = [0.1, 1, 0.1+wth/20]  # [0.1, 1, 0.2-0.3]
    preds_nonUS_f = signal.convolve(preds_nonUS, MA3pt_filter, mode='same', method='auto')

    weighted_vote_result = []
    for ith in range(y.shape[0]//45):  # iterating through each spelling number consisting of 45 flashes
        # votes = y_val[ith*45:(ith+1)*45][preds_nonUS[ith*45:(ith+1)*45]==1]
        # modes = pd.DataFrame(votes).value_counts().index.tolist()
        P_weighted = []
        first45_preds_f = preds_nonUS_f[ith*45:(ith+1)*45]
        first45_y_val = y_val[ith*45:(ith+1)*45]
        for jth in range(9):
            num = jth+1
            P_weighted += [np.sum(first45_preds_f[first45_y_val == num])]
        # print(P_weighted)
        # print(votes)
        # print(modes[0][0])
        # vote_result += [modes[0][0]]
        weighted_vote_result += [max(range(len(P_weighted)), key=P_weighted.__getitem__)+1]
        # index_max = max(range(len(P_weighted)), key=P_weighted.__getitem__)
    print("correct percentage (for weighted):", (sum(np.array(num_markers) == np.array(weighted_vote_result))/27))
    return


def show_vote_histogram(y, y_val, preds_nonUS):
    vote_result = []
    for ith in range(y.shape[0] // 45):
        votes = y_val[ith * 45: (ith + 1) * 45][preds_nonUS[ith * 45: (ith + 1) * 45] == 1]
        modes = pd.DataFrame(votes).value_counts().index.tolist()
        vote_result += [modes[0][0]]
        plt.figure()
        plt.hist(votes)
