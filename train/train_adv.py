import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE, make_deterministic
from auxiliary.utils import log_metrics
from classes.adv.ModelAdvConfFC4 import ModelAdvConfFC4
from classes.data.ColorCheckerDataset import ColorCheckerDataset
from classes.training.Evaluator import Evaluator
from classes.training.LossTracker import LossTracker

RANDOM_SEED = 0
EPOCHS = 1000
BATCH_SIZE = 1
LEARNING_RATE = 0.0003
ADV_LAMBDA = 0.05
FOLD_NUM = 0

PATH_TO_BASE_MODEL = os.path.join("trained_models", "adv", "base", "fold_{}".format(FOLD_NUM))


def main(opt):
    fold_num, epochs, batch_size, lr, adv_lambda = opt.fold_num, opt.epochs, opt.batch_size, opt.lr, opt.adv_lambda

    path_to_log = os.path.join("logs", "adv_{}_fold_{}_{}".format(str(opt.adv_lambda), str(fold_num), str(time.time())))
    os.makedirs(path_to_log, exist_ok=True)

    path_to_vis = os.path.join(path_to_log, "vis")
    os.makedirs(path_to_vis, exist_ok=True)

    path_to_metrics = os.path.join(path_to_log, "metrics.csv")

    model = ModelAdvConfFC4(adv_lambda)
    print("\n Loading base model at: {} \n".format(PATH_TO_BASE_MODEL))
    model.load(PATH_TO_BASE_MODEL)
    print("\n Saving base model at: {} \n".format(path_to_log))
    model.save(path_to_log)

    model.print_network()
    model.log_network(path_to_log)
    model.set_optimizer(lr)

    training_set = ColorCheckerDataset(train=True, folds_num=fold_num)
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)
    print("\n Training set size ... : {}".format(len(training_set)))

    test_set = ColorCheckerDataset(train=False, folds_num=fold_num)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=True)
    print(" Test set size ....... : {}\n".format(len(test_set)))

    print("\n**************************************************************")
    print("\t\t\t Training Adversary FC4 - Fold {}".format(fold_num))
    print("**************************************************************\n")

    evaluator_base, evaluator_adv = Evaluator(), Evaluator()
    best_val_loss, best_metrics = 100.0, evaluator_base.get_best_metrics()
    train_loss, val_loss = LossTracker(), LossTracker()

    for epoch in range(epochs):

        model.train_mode()
        train_loss.reset()
        start = time.time()

        for i, (img, label, filename) in enumerate(training_loader):
            img, label = img.to(DEVICE), label.to(DEVICE)
            (pred_base, _, conf_base), (pred_adv, _, conf_adv) = model.predict(img)
            loss, losses = model.optimize(pred_base, pred_adv, conf_base, conf_adv)
            train_loss.update(loss)

            err_base = model.get_loss(pred_base, label).item()
            err_adv = model.get_loss(pred_adv, label).item()

            if i % 5 == 0:
                loss_log = " - ".join(["{}: {:.4f}".format(lt, lv.item()) for lt, lv in losses.items()])
                print("[ Epoch: {}/{} - Batch: {} ] "
                      "| Loss: [ train: {:.4f} - {} ] "
                      "| Error: [ base: {:.4f} - adv: {:.4f} ]"
                      .format(epoch + 1, epochs, i, loss, loss_log, err_base, err_adv))

        if epoch % 50 == 0:
            path_to_save = os.path.join(path_to_vis, "epoch_{}".format(epoch))
            print("\n Saving vis at: {} \n".format(path_to_save))
            model.save_vis(img, conf_base, conf_adv, path_to_save)

        train_time = time.time() - start

        val_loss.reset()
        start = time.time()

        if epoch % 5 == 0:
            evaluator_base.reset_errors()
            evaluator_adv.reset_errors()
            model.evaluation_mode()

            print("\n--------------------------------------------------------------")
            print("\t\t\t Validation")
            print("--------------------------------------------------------------\n")

            with torch.no_grad():
                for i, (img, label, _) in enumerate(test_loader):
                    img, label = img.to(DEVICE), label.to(DEVICE)
                    (pred_base, _, conf_base), (pred_adv, _, conf_adv) = model.predict(img)
                    loss, losses = model.get_losses(conf_base, conf_adv, pred_base, pred_adv)
                    loss = loss.item()
                    val_loss.update(loss)

                    err_base = model.get_loss(pred_base, label).item()
                    err_adv = model.get_loss(pred_adv, label).item()

                    evaluator_base.add_error(err_base)
                    evaluator_adv.add_error(err_adv)

                    if i % 5 == 0:
                        loss_log = " - ".join(["{}: {:.4f}".format(lt, lv.item()) for lt, lv in losses.items()])
                        print("[ Epoch: {}/{} - Batch: {} ] "
                              "| Loss: [ val: {:.4f} - {} ] "
                              "| Error: [ base: {:.4f} - adv: {:.4f} ]"
                              .format(epoch + 1, epochs, i, loss, loss_log, err_base, err_adv))

            print("\n--------------------------------------------------------------\n")

        val_time = time.time() - start

        metrics_base, metrics_adv = evaluator_base.compute_metrics(), evaluator_adv.compute_metrics()
        print("\n********************************************************************")
        print(" Train Time ........ : {:.4f}".format(train_time))
        print(" Train Loss ........ : {:.4f}".format(train_loss.avg))
        if val_time > 0.1:
            print("....................................................................")
            print(" Val Time ......... : {:.4f}".format(val_time))
            print(" Val Loss ......... : {:.4f}".format(val_loss.avg))
            print("....................................................................")
            print(" Mean ......... : {:.4f} (Best: {:.4f} - Base: {:.4f})"
                  .format(metrics_adv["mean"], best_metrics["mean"], metrics_base["mean"]))
            print(" Median ....... : {:.4f} (Best: {:.4f} - Base: {:.4f}))"
                  .format(metrics_adv["median"], best_metrics["median"], metrics_base["median"]))
            print(" Trimean ...... : {:.4f} (Best: {:.4f} - Base: {:.4f}))"
                  .format(metrics_adv["trimean"], best_metrics["trimean"], metrics_base["trimean"]))
            print(" Best 25% ..... : {:.4f} (Best: {:.4f} - Base: {:.4f}))"
                  .format(metrics_adv["bst25"], best_metrics["bst25"], metrics_base["bst25"]))
            print(" Worst 25% .... : {:.4f} (Best: {:.4f} - Base: {:.4f}))"
                  .format(metrics_adv["wst25"], best_metrics["wst25"], metrics_base["wst25"]))
            print(" Worst 5% ..... : {:.4f} (Best: {:.4f} - Base: {:.4f}))"
                  .format(metrics_adv["wst5"], best_metrics["wst5"], metrics_base["wst5"]))
        print("********************************************************************\n")

        if 0 < val_loss.avg < best_val_loss:
            best_val_loss = val_loss.avg
            best_metrics = evaluator_adv.update_best_metrics()
            print("Saving new best models... \n")
            model.save_adv(path_to_log)

        log_metrics(train_loss.avg, val_loss.avg, metrics_adv, best_metrics, path_to_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_num", type=int, default=FOLD_NUM)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--adv_lambda', type=float, default=ADV_LAMBDA)
    opt = parser.parse_args()
    make_deterministic(opt.random_seed)

    print("\n *** Training configuration ***")
    print("\t Fold num ........ : {}".format(opt.fold_num))
    print("\t Epochs .......... : {}".format(opt.epochs))
    print("\t Batch size ...... : {}".format(opt.batch_size))
    print("\t Learning rate ... : {}".format(opt.lr))
    print("\t Random seed ..... : {}".format(opt.random_seed))
    print("\t Adv lambda ...... : {}".format(opt.adv_lambda))

    main(opt)
