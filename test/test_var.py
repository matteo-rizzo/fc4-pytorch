import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.utils.data
from torch.utils.data import DataLoader

from ModelAdvConfFC4 import ModelAdvConfFC4
from auxiliary.settings import DEVICE, make_deterministic
from auxiliary.utils import jsd, scale
from classes.data.ColorCheckerDataset import ColorCheckerDataset
from classes.training.Evaluator import Evaluator

NUM_FOLD = 0
PATH_TO_PRETRAINED = os.path.join("trained_models")


def evaluate(model: ModelAdvConfFC4, dataloader: DataLoader):
    evaluator, evaluator_adv = Evaluator(), Evaluator()
    adv_conf_values, jsd_values = [], []

    model.evaluation_mode()

    with torch.no_grad():
        for i, (img, label, file_name) in enumerate(dataloader):
            img, label = img.to(DEVICE), label.to(DEVICE)
            (pred, _, confidence), (pred_adv, _, confidence_adv) = model.predict(img)

            loss = model.get_loss(pred, label)
            evaluator.add_error(loss.item())
            loss_adv = model.get_loss(pred_adv, label)
            evaluator_adv.add_error(loss_adv.item())

            confidence = torch.flatten(scale(confidence)).numpy()
            confidence_adv = torch.flatten(scale(confidence_adv)).numpy()

            adv_conf_values.append(float(abs(np.mean(confidence_adv) - np.mean(confidence))))
            jsd_values.append(float(jsd(confidence, confidence_adv)))

            print('\t - Input: {} - Batch: {} | Loss: {:.4f} - Loss Adv: {:.4f}'
                  .format(file_name[0], i + 1, loss.item(), loss_adv.item()))

        metrics_base, metrics_adv = evaluator.compute_metrics(), evaluator_adv.compute_metrics()
        print(" \n Mean ........ : [ Base: {:.4f} | Adv: {:.4f} ]".format(metrics_base["mean"], metrics_adv["mean"]))
        print(" Median ...... : [ Base: {:.4f} | Adv: {:.4f} ]".format(metrics_base["median"], metrics_adv["median"]))
        print(" Trimean ..... : [ Base: {:.4f} | Adv: {:.4f} ]".format(metrics_base["trimean"], metrics_adv["trimean"]))
        print(" Best 25% .... : [ Base: {:.4f} | Adv: {:.4f} ]".format(metrics_base["bst25"], metrics_adv["bst25"]))
        print(" Worst 25% ... : [ Base: {:.4f} | Adv: {:.4f} ]".format(metrics_base["wst25"], metrics_adv["wst25"]))
        print(" Worst 5% .... : [ Base: {:.4f} | Adv: {:.4f} ] \n".format(metrics_base["wst5"], metrics_adv["wst5"]))

        return adv_conf_values, jsd_values


def main():
    test_set = ColorCheckerDataset(train=False, folds_num=NUM_FOLD)
    dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=20)

    print("\n *** Fold {} - Test set size: {} *** \n".format(NUM_FOLD, len(test_set)))

    conf_seeds, conf_values, jsd_seeds, jsd_values = [], [], [], []
    base_path_to_pretrained = os.path.join(PATH_TO_PRETRAINED, "variance", "fold_{}".format(NUM_FOLD))
    model_dirs = sorted(os.listdir(base_path_to_pretrained))
    for i, model_dir in enumerate(model_dirs):
        s = model_dir.split("_")[1]
        print("\n -> [ {}/{} ] Seed: {}".format(i + 1, len(model_dirs), s))

        path_to_pretrained = os.path.join(base_path_to_pretrained, model_dir)
        print("\n Using pretrained model stored at: {} \n".format(path_to_pretrained))

        model = ModelAdvConfFC4()
        model.load(path_to_pretrained)

        conf_seed, jsd_seed = evaluate(model, dataloader)
        conf_seeds.append(conf_seed)
        conf_values += conf_seed
        jsd_seeds.append(jsd_seed)
        jsd_values += jsd_seed
        print("\n AVG: {:.4f} - STD DEV: {:.4f} \n".format(np.mean(jsd_seed), np.std(jsd_seed)))
        sns.kdeplot(jsd_seed, conf_seed, shade=True, thresh=0.05, alpha=0.5)

    path_to_save = os.path.join("test", "results")
    os.makedirs(path_to_save, exist_ok=True)

    plt.xlabel("JSD")
    plt.ylabel("Average Confidence Difference")
    plt.title("Seeds Variance")
    plt.savefig(os.path.join(path_to_save, "var_jsd_conf_seeds_{}.png".format(time.time())), bbox_inches='tight')
    plt.show()
    plt.clf()

    plt.xlabel("JSD")
    plt.ylabel("Average Confidence Difference")
    plt.title("Seeds Variance [overall]")
    sns.kdeplot(x=jsd_values, y=conf_values, shade=False, thresh=0.05, alpha=0.5)
    plt.savefig(os.path.join(path_to_save, "var_jsd_conf_{}.png".format(time.time())), bbox_inches='tight')
    plt.show()
    plt.clf()

    for conf_seed in conf_seeds:
        sns.kdeplot(conf_seed, shade=True, thresh=0.05, alpha=0.5)
    plt.title("Average Confidence Difference")
    plt.savefig(os.path.join(path_to_save, "var_conf_seeds_{}.png".format(time.time())), bbox_inches='tight')
    plt.show()
    plt.clf()

    sns.kdeplot(conf_values, shade=False, thresh=0.05)
    plt.title("Average Confidence Difference [overall]")
    plt.savefig(os.path.join(path_to_save, "var_conf_{}.png".format(time.time())), bbox_inches='tight')
    plt.show()
    plt.clf()

    for jsd_seed in jsd_seeds:
        sns.kdeplot(jsd_seed, shade=True, thresh=0.05, alpha=0.5)
    plt.title("JSD (divergence)")
    plt.savefig(os.path.join(path_to_save, "var_jsd_seeds_{}.png".format(time.time())), bbox_inches='tight')
    plt.show()
    plt.clf()

    sns.kdeplot(jsd_values, shade=False, thresh=0.05)
    plt.title("JSD (divergence) [overall]")
    plt.savefig(os.path.join(path_to_save, "var_jsd_{}.png".format(time.time())), bbox_inches='tight')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    make_deterministic(seed=1)
    main()
