import os
from time import time

import numpy as np
import torch.utils.data

from auxiliary.settings import DEVICE
from classes.core.Evaluator import Evaluator
from classes.data.ColorCheckerDataset import ColorCheckerDataset
from classes.fc4.ModelFC4 import ModelFC4

"""
* FC4 using confidence-weighted pooling (fc_cwp):

Fold	Mean		Median		Trimean		Best 25%	Worst 25%	Worst 5%
0	    1.73		1.47		1.50		0.50		3.53		4.20
1	    2.11		1.54		1.66		0.43		4.87		5.89
2	    1.92		1.45		1.52		0.52		4.22		5.66
Avg	    1.92		1.49		1.56		0.48		4.21		5.25
StdDev	0.19		0.05		0.09		0.05		0.67		0.92

* FC4 using summation pooling (fc_sum):

Fold	Mean		Median		Trimean		Best 25%	Worst 25%	Worst 5%	
0	    1.68        1.20	    1.35    	0.40	    3.71	    4.25
1	    2.11	    1.62	    1.68	    0.51	    4.74	    5.78
2	    1.79	    1.24	    1.35	    0.38	    4.21	    5.60
Avg	    1.86	    1.35	    1.46	    0.43	    4.22	    5.21
StdDev  0.22	    0.23	    0.19	    0.07	    0.52	    0.84
"""

MODEL_TYPE = "baseline/fc4_cwp"
SAVE_PRED = False
SAVE_CONF = False
USE_TRAINING_SET = False


def main():
    evaluator = Evaluator()
    model = ModelFC4()
    path_to_pred, path_to_pred_fold = None, None
    path_to_conf, path_to_conf_fold = None, None

    if SAVE_PRED:
        path_to_pred = os.path.join("test", "pred", "{}_{}".format("train" if USE_TRAINING_SET else "test", time()))

    if SAVE_CONF:
        path_to_conf = os.path.join("test", "conf", "{}_{}".format("train" if USE_TRAINING_SET else "test", time()))

    for num_fold in range(3):
        fold_evaluator = Evaluator()
        test_set = ColorCheckerDataset(train=USE_TRAINING_SET, folds_num=num_fold)
        dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=16)

        path_to_pretrained = os.path.join("trained_models", MODEL_TYPE, "fold_{}".format(num_fold))
        model.load(path_to_pretrained)
        model.evaluation_mode()

        if SAVE_PRED:
            path_to_pred_fold = os.path.join(path_to_pred, "fold_{}".format(num_fold))
            os.makedirs(path_to_pred_fold)

        if SAVE_CONF:
            path_to_conf_fold = os.path.join(path_to_conf, "fold_{}".format(num_fold))
            os.makedirs(path_to_conf_fold)

        print("\n *** FOLD {} *** \n".format(num_fold))
        print(" * Test set size: {}".format(len(test_set)))
        print(" * Using trained model stored at: {} \n".format(path_to_pretrained))

        with torch.no_grad():
            for i, (img, label, file_name) in enumerate(dataloader):
                img, label = img.to(DEVICE), label.to(DEVICE)
                pred, _, conf = model.predict(img, return_steps=True)
                loss = model.get_loss(pred, label)
                fold_evaluator.add_error(loss.item())
                evaluator.add_error(loss.item())
                print('\t - Input: {} - Batch: {} | Loss: {:f}'.format(file_name[0], i, loss.item()))
                if SAVE_PRED:
                    np.save(os.path.join(path_to_pred_fold, file_name[0]), pred)
                if SAVE_CONF:
                    np.save(os.path.join(path_to_conf_fold, file_name[0]), conf)

        metrics = fold_evaluator.compute_metrics()
        print("\n Mean ............ : {:.4f}".format(metrics["mean"]))
        print(" Median .......... : {:.4f}".format(metrics["median"]))
        print(" Trimean ......... : {:.4f}".format(metrics["trimean"]))
        print(" Best 25% ........ : {:.4f}".format(metrics["bst25"]))
        print(" Worst 25% ....... : {:.4f}".format(metrics["wst25"]))
        print(" Percentile 95 ... : {:.4f} \n".format(metrics["wst5"]))

    print("\n *** AVERAGE ACROSS FOLDS *** \n")
    metrics = evaluator.compute_metrics()
    print("\n Mean ............ : {:.4f}".format(metrics["mean"]))
    print(" Median .......... : {:.4f}".format(metrics["median"]))
    print(" Trimean ......... : {:.4f}".format(metrics["trimean"]))
    print(" Best 25% ........ : {:.4f}".format(metrics["bst25"]))
    print(" Worst 25% ....... : {:.4f}".format(metrics["wst25"]))
    print(" Percentile 95 ... : {:.4f} \n".format(metrics["wst5"]))


if __name__ == '__main__':
    main()
