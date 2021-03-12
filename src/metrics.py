def hit_and_auc(rankedlist, test_item, k):
    hits_k = [(idx, val) for idx, val in enumerate(rankedlist[:k])
              if int(val) == int(test_item)]
    hits_all = [(idx, val) for idx, val in enumerate(rankedlist)
                if int(val) == int(test_item)]

    max_num = len(rankedlist) - 1
    auc = 1.0 * (max_num - hits_all[0][0]) / \
        max_num if len(hits_all) > 0 else 0
    return len(hits_k), auc
