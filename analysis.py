import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def vote_count(y, y_val, preds_nonUS):
    ### IMPORTANT: DO YOU NEED THE RANDOM SEQUENCE, OR CAN YOU JUST VOTE WITH YES/NO?
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
        votes = y_val[ith * 45 : (ith + 1) * 45][preds_nonUS[ith * 45 : (ith + 1) * 45] == 1]
        modes = pd.DataFrame(votes).value_counts().index.tolist()
        vote_result += [modes[0][0]]
    print("vote_result", vote_result)
    print("vote_truth ", num_markers)
    print("correct percentage:", (sum(np.array(num_markers) == np.array(vote_result)) / 27))


def show_vote_histogram(y, y_val, preds_nonUS):
    vote_result = []
    for ith in range(y.shape[0] // 45):
        votes = y_val[ith * 45 : (ith + 1) * 45][preds_nonUS[ith * 45 : (ith + 1) * 45] == 1]
        modes = pd.DataFrame(votes).value_counts().index.tolist()
        vote_result += [modes[0][0]]
        plt.figure()
        plt.hist(votes)
