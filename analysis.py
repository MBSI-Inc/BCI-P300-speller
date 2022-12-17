import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal

N_CHAR = 9  # Number of available characters for spelling
N_REPEAT = 5
N_FLASH_PER_CHAR = N_CHAR * N_REPEAT


def vote_count(y, y_val, preds_nonUS):
    # IMPORTANT: DO YOU NEED THE RANDOM SEQUENCE, OR CAN YOU JUST VOTE WITH YES/NO?
    print(type(y_val))
    print(y_val.shape)
    print(y_val[preds_nonUS == 1][:N_CHAR+1])
    print(y_val[y == 1][:3])
    tmp = y_val[y == 1]
    print(tmp.shape)

    preds_set_hit = np.zeros(y.shape[0] // N_CHAR)
    for i in range(N_CHAR):
        # rows = hit_seq[i::12]
        rows = preds_nonUS[i::N_CHAR]
        # set_hit = np.vstack((set_hit, rows))
        preds_set_hit = np.vstack((preds_set_hit, rows))
    print(np.shape(preds_set_hit))
    plt.hist(sum(preds_set_hit[1:, :]))  # TELLS HOW MANY HITS PER 9 FLASHES, SHOULD HAVE 1 ONLY
    preds_set_hit = np.transpose(preds_set_hit[1:, :])
    print(np.shape(preds_set_hit))
    print(preds_set_hit[0:N_CHAR+1])


def print_predict_and_truth(y, y_val, preds_nonUS, num_markers):
    n_spelled = len(num_markers)
    vote_result = []
    n_group = y.shape[0] // N_FLASH_PER_CHAR

    for i in range(n_group):
        current_group_y_val = y_val[i * N_FLASH_PER_CHAR: (i + 1) * N_FLASH_PER_CHAR]
        current_group_preds_nonUS = preds_nonUS[i * N_FLASH_PER_CHAR: (i + 1) * N_FLASH_PER_CHAR]
        votes = current_group_y_val[current_group_preds_nonUS == 1]
        modes = pd.DataFrame(votes).value_counts().index.tolist()
        vote_result += [modes[0][0]]
    print("vote_result", vote_result)
    print("vote_truth ", num_markers)
    print("correct percentage:", (sum(np.array(num_markers) == np.array(vote_result)) / n_spelled))

    wth = 1
    MA3pt_filter = [0.1, 1, 0.2]  # [0.1, 1, 0.2-0.3]
    preds_nonUS_f = signal.convolve(preds_nonUS, MA3pt_filter, mode='same', method='auto')

    weighted_vote_result = []
    for i in range(n_group):  # iterating through each spelling number consisting of 45 flashes
        P_weighted = []
        current_group_y_val = y_val[i * N_FLASH_PER_CHAR: (i + 1) * N_FLASH_PER_CHAR]
        current_group_preds_nonUS_f = preds_nonUS_f[i * N_FLASH_PER_CHAR: (i + 1) * N_FLASH_PER_CHAR]
        for j in range(N_CHAR):
            num = j+1
            P_weighted += [np.sum(current_group_preds_nonUS_f[current_group_y_val == num])]
        weighted_vote_result += [max(range(len(P_weighted)), key=P_weighted.__getitem__)+1]
    # TODO: Why divide by 27
    print("correct percentage (for weighted):", (sum(np.array(num_markers) == np.array(weighted_vote_result))/n_spelled))
    return


def show_vote_histogram(y, y_val, preds_nonUS):
    vote_result = []
    n_group = y.shape[0] // N_FLASH_PER_CHAR

    for ith in range(n_group):
        cur_group_y_val = y_val[ith * N_FLASH_PER_CHAR: (ith + 1) * N_FLASH_PER_CHAR]
        cur_group_preds_nonUS = preds_nonUS[ith * N_FLASH_PER_CHAR: (ith + 1) * N_FLASH_PER_CHAR]
        votes = cur_group_y_val[cur_group_preds_nonUS == 1]
        modes = pd.DataFrame(votes).value_counts().index.tolist()
        vote_result += [modes[0][0]]
        plt.figure()
        plt.hist(votes)
