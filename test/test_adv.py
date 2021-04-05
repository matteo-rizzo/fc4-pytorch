import os

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE
from auxiliary.utils import angular_error, jsd, scale
from classes.data.ColorCheckerDataset import ColorCheckerDataset
from classes.fc4.ModelAdvConfFC4 import ModelAdvConfFC4
from classes.training.Evaluator import Evaluator

NUM_FOLD = 0
PATH_TO_PRETRAINED = os.path.join("trained_models")


def evaluate(model: ModelAdvConfFC4, dataloader: DataLoader):
    evaluator, evaluator_adv = Evaluator(), Evaluator()
    eval_data = {"preds": [], "conf": [], "preds_adv": [], "conf_adv": []}

    model.evaluation_mode()

    with torch.no_grad():
        for i, (img, label, file_name) in enumerate(dataloader):
            img, label = img.to(DEVICE), label.to(DEVICE)
            (pred, _, confidence), (pred_adv, _, confidence_adv) = model.predict(img)

            loss = model.get_angular_loss(pred, label)
            evaluator.add_error(loss.item())
            eval_data["preds"].append(pred)
            eval_data["conf"].append(torch.flatten(scale(confidence)).numpy())

            loss_adv = model.get_angular_loss(pred_adv, label)
            evaluator_adv.add_error(loss_adv.item())
            eval_data["preds_adv"].append(pred_adv)
            eval_data["conf_adv"].append(torch.flatten(scale(confidence_adv)).numpy())

            print('\t - Input: {} - Batch: {} | Loss: {:.4f} - Loss Adv: {:.4f}'
                  .format(file_name[0], i + 1, loss.item(), loss_adv.item()))

        metrics_base, metrics_adv = evaluator.compute_metrics(), evaluator_adv.compute_metrics()
        print(" \n Mean ........ : [ Base: {:.4f} | Adv: {:.4f} ]".format(metrics_base["mean"], metrics_adv["mean"]))
        print(" Median ...... : [ Base: {:.4f} | Adv: {:.4f} ]".format(metrics_base["median"], metrics_adv["median"]))
        print(" Trimean ..... : [ Base: {:.4f} | Adv: {:.4f} ]".format(metrics_base["trimean"], metrics_adv["trimean"]))
        print(" Best 25% .... : [ Base: {:.4f} | Adv: {:.4f} ]".format(metrics_base["bst25"], metrics_adv["bst25"]))
        print(" Worst 25% ... : [ Base: {:.4f} | Adv: {:.4f} ]".format(metrics_base["wst25"], metrics_adv["wst25"]))
        print(" Worst 5% .... : [ Base: {:.4f} | Adv: {:.4f} ] \n".format(metrics_base["wst5"], metrics_adv["wst5"]))

        err = [angular_error(pb, pa) for pb, pa in zip(eval_data["preds"], eval_data["preds_adv"])]
        div = [jsd(cb, ca) for cb, ca in zip(eval_data["conf"], eval_data["conf_adv"])]

        return np.mean(err), np.mean(div)


def main():
    adv_data, var_data = {"errors": [], "divergences": []}, {"errors": [], "divergences": []}
    test_set = ColorCheckerDataset(train=False, folds_num=NUM_FOLD)
    dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=20)

    print("\n *** Fold {} - Test set size: {} *** \n".format(NUM_FOLD, len(test_set)))

    base_path_to_pretrained = os.path.join(PATH_TO_PRETRAINED, "adv", "fold_{}".format(NUM_FOLD))
    for model_dir in sorted(os.listdir(base_path_to_pretrained)):
        print("\n -> Lambda: {}".format("0." + model_dir.split("L")[1]))

        path_to_pretrained = os.path.join(base_path_to_pretrained, model_dir)
        print("\n Using pretrained model stored at: {} \n".format(path_to_pretrained))

        model = ModelAdvConfFC4()
        model.load(path_to_pretrained)

        err, div = evaluate(model, dataloader)
        adv_data["errors"].append(err)
        adv_data["divergences"].append(div)

    base_path_to_pretrained = os.path.join(PATH_TO_PRETRAINED, "variance", "fold_{}".format(NUM_FOLD))
    for model_dir in sorted(os.listdir(base_path_to_pretrained)):
        print("\n -> Seed: {}".format(model_dir.split("_")[1]))

        path_to_pretrained = os.path.join(base_path_to_pretrained, model_dir)
        print("\n Using pretrained model stored at: {} \n".format(path_to_pretrained))

        model = ModelAdvConfFC4()
        model.load(path_to_pretrained)

        err, div = evaluate(model, dataloader)
        var_data["errors"].append(err)
        var_data["divergences"].append(div)

    plt.plot(adv_data["divergences"], adv_data["errors"], linestyle='--', marker='o', color='r')
    plt.scatter(var_data["divergences"], var_data["errors"], marker='^', color='g')
    plt.xlabel("Confidence JSD")
    plt.ylabel("Predictions AE")
    plt.savefig(os.path.join("test", "adv.png"), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
