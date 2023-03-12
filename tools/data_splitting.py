from sklearn.model_selection import KFold
import numpy as np
import os

def get_split_deterministic(all_keys, fold=0, num_splits=5, random_state=12345):
    """
    Splits a list of patient identifiers (or numbers) into number splits folds and returns the split for fold fold.
    :param all_keys:
    :param fold:
    :param num_splits:
    :param random_state:
    :return:
    """
    all_keys_sorted = np.sort(np.array(all_keys))
    splits = KFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    for i, (train_idx, test_idx) in enumerate(splits.split(all_keys_sorted)):
        if i == fold:
            train_keys = np.array(all_keys_sorted)[train_idx]
            val_keys = np.array(all_keys_sorted)[test_idx]
            break
    return train_keys.tolist(), val_keys.tolist()

if __name__ == '__main__':
    path=r'C:\Users\Administrator\Desktop\rightcrop'
    list=os.listdir(path)
    train_keys, val_keys=get_split_deterministic(all_keys=list,fold=0,num_splits=2,random_state=12345)
    train_list = [os.path.join(path,x) for x in train_keys]
    val_list = [os.path.join(path,x) for x in val_keys]
    print(train_list)
