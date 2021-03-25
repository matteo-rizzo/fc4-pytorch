import os

import matplotlib.pyplot as plt
import torch.utils.data

from auxiliary.settings import DEVICE
from auxiliary.utils import angular_error, jsd, scale
from classes.data.ColorCheckerDataset import ColorCheckerDataset
from classes.fc4.ModelAdvConfFC4 import ModelAdvConfFC4
from classes.training.Evaluator import Evaluator

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

MODEL_TYPE = "fc4_adv"


def main():
    evaluator, evaluator_adv = Evaluator(), Evaluator()
    model = ModelAdvConfFC4()

    for num_fold in range(1):
        fold_evaluator, fold_evaluator_adv = Evaluator(), Evaluator()
        fold_eval_data = {"preds": [], "conf": [], "preds_adv": [], "conf_adv": []}

        test_set = ColorCheckerDataset(train=False, folds_num=num_fold)
        dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=20)

        path_to_pretrained = os.path.join("../trained_models", MODEL_TYPE, "fold_{}".format(num_fold))
        model.load(path_to_pretrained)
        model.evaluation_mode()

        print("\n *** FOLD {} *** \n".format(num_fold))
        print(" * Test set size: {}".format(len(test_set)))
        print(" * Using pretrained model stored at: {} \n".format(path_to_pretrained))

        with torch.no_grad():
            for i, (img, label, file_name) in enumerate(dataloader):
                img, label = img.to(DEVICE), label.to(DEVICE)
                (pred, _, confidence), (pred_adv, _, confidence_adv) = model.predict(img)

                loss = model.get_angular_loss(pred, label)
                fold_evaluator.add_error(loss.item())
                evaluator.add_error(loss.item())
                fold_eval_data["preds"].append(pred)
                fold_eval_data["conf"].append(torch.flatten(scale(confidence)).numpy())

                loss_adv = model.get_angular_loss(pred_adv, label)
                fold_evaluator_adv.add_error(loss_adv.item())
                evaluator_adv.add_error(loss_adv.item())
                fold_eval_data["preds_adv"].append(pred_adv)
                fold_eval_data["conf_adv"].append(torch.flatten(scale(confidence_adv)).numpy())

                print('\t - Input: {} - Batch: {} | Loss: {:.4f} - Loss Adv: {:.4f}'
                      .format(file_name[0], i + 1, loss.item(), loss_adv.item()))

        metrics, metrics_adv = fold_evaluator.compute_metrics(), fold_evaluator_adv.compute_metrics()
        print("\n Mean ...... : [ Base: {:.4f} | Adv: {:.4f} ]".format(metrics["mean"], metrics_adv["mean"]))
        print(" Median ...... : [ Base: {:.4f} | Adv: {:.4f} ]".format(metrics["median"], metrics_adv["median"]))
        print(" Trimean ..... : [ Base: {:.4f} | Adv: {:.4f} ]".format(metrics["trimean"], metrics_adv["trimean"]))
        print(" Best 25% .... : [ Base: {:.4f} | Adv: {:.4f} ]".format(metrics["bst25"], metrics_adv["bst25"]))
        print(" Worst 25% ... : [ Base: {:.4f} | Adv: {:.4f} ]".format(metrics["wst25"], metrics_adv["wst25"]))
        print(" Worst 5% .... : [ Base: {:.4f} | Adv: {:.4f} ] \n".format(metrics["wst5"], metrics_adv["wst5"]))

        errors = [angular_error(pb, pa) for pb, pa in zip(fold_eval_data["preds"], fold_eval_data["preds_adv"])]
        div = [jsd(cb, ca) for cb, ca in zip(fold_eval_data["conf"], fold_eval_data["conf_adv"])]
        div, errors = zip(*sorted(zip(div, errors), key=lambda x: x[0]))

        plt.plot(div, errors)
        plt.xlabel("Confidence JSD")
        plt.ylabel("Angular error")
        plt.show()

    print("\n *** AVERAGE ACROSS FOLDS *** \n")
    metrics, metrics_adv = evaluator.compute_metrics(), evaluator_adv.compute_metrics()
    print("\n Mean ...... : [ Base: {:.4f} | Adv: {:.4f} ]".format(metrics["mean"], metrics_adv["mean"]))
    print(" Median ...... : [ Base: {:.4f} | Adv: {:.4f} ]".format(metrics["median"], metrics_adv["median"]))
    print(" Trimean ..... : [ Base: {:.4f} | Adv: {:.4f} ]".format(metrics["trimean"], metrics_adv["trimean"]))
    print(" Best 25% .... : [ Base: {:.4f} | Adv: {:.4f} ]".format(metrics["bst25"], metrics_adv["bst25"]))
    print(" Worst 25% ... : [ Base: {:.4f} | Adv: {:.4f} ]".format(metrics["wst25"], metrics_adv["wst25"]))
    print(" Worst 5% .... : [ Base: {:.4f} | Adv: {:.4f} ] \n".format(metrics["wst5"], metrics_adv["wst5"]))


if __name__ == '__main__':
    main()
