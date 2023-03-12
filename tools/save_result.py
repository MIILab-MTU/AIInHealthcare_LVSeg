import pandas as pd
import os
def save_train_result(savepath, epoch,epochs_dice,epochs_loss):
    train_result = []
    train_result.append(epochs_dice)
    train_result.append(epochs_loss)
    result = pd.DataFrame(train_result)
    index_name = {0: "epochs_dice", 1: "epochs_loss"}
    result = result.rename(index=index_name, inplace=False)
    result.to_excel(os.path.join(savepath, 'train_epoch_{}.xlsx'.format(epoch)))


def save_val_result(savepath, epoch,epochs_dice):
    val_result = []
    val_result.append(epochs_dice)
    result = pd.DataFrame(val_result)
    index_name = {0: "epochs_dice"}
    result = result.rename(index=index_name, inplace=False)
    result.to_excel(os.path.join(savepath, 'val_epoch_{}.xlsx'.format(epoch)))
