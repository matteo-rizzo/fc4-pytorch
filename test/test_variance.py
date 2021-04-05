import os

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data

from auxiliary.settings import DEVICE
from auxiliary.utils import angular_error, jsd, scale
from classes.data.ColorCheckerDataset import ColorCheckerDataset
from classes.fc4.ModelAdvConfFC4 import ModelAdvConfFC4
from classes.training.Evaluator import Evaluator

NUM_FOLD = 0
PATH_TO_PRETRAINED = os.path.join("trained_models", "variance", "fold_{}".format(NUM_FOLD))


def main():
    errors, divergences = [], []
    evaluator, evaluator_adv = Evaluator(), Evaluator()
    for seed in sorted(os.listdir(PATH_TO_PRETRAINED)):
        fold_evaluator, fold_evaluator_adv = Evaluator(), Evaluator()
        fold_eval_data = {"preds": [], "conf": [], "preds_adv": [], "conf_adv": []}

        test_set = ColorCheckerDataset(train=False, folds_num=NUM_FOLD)
        dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=20)

        path_to_pretrained = os.path.join(PATH_TO_PRETRAINED, seed)
        model = ModelAdvConfFC4()
        model.load(path_to_pretrained)
        model.evaluation_mode()

        seed = int(seed.split("_")[1])
        print("\n *** FOLD {} - SEED: {} *** \n".format(NUM_FOLD, seed))
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

        err = [angular_error(pb, pa) for pb, pa in zip(fold_eval_data["preds"], fold_eval_data["preds_adv"])]
        errors.append(np.mean(err))

        div = [jsd(cb, ca) for cb, ca in zip(fold_eval_data["conf"], fold_eval_data["conf_adv"])]
        divergences.append(np.mean(div))

    print("div: {} \n err: {}".format(divergences, errors))

    plt.scatter(divergences, errors, 'g^')
    plt.xlabel("Confidence JSD")
    plt.ylabel("Angular error")
    plt.show()
    plt.savefig(os.path.join("test", "seeds_variance.png"), bbox_inches='tight')

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
