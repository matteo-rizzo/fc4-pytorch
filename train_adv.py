import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE
from auxiliary.utils import print_metrics, log_metrics
from classes.data.ColorCheckerDataset import ColorCheckerDataset
from classes.fc4.ModelAdvConfFC4 import ModelAdvConfFC4
from classes.training.Evaluator import Evaluator
from classes.training.LossTracker import LossTracker

EPOCHS = 2000
BATCH_SIZE = 1
LEARNING_RATE = 0.0003

# Which of the 3 folds should be processed (either 0, 1 or 2)
FOLD_NUM = 0

# The subset of test images to be monitored (set to empty list to skip saving visualizations and speed up training)
# TEST_VIS_IMG = ["IMG_0753", "IMG_0438", "IMG_0397"]
TEST_VIS_IMG = []

RELOAD_CHECKPOINT = False
PATH_TO_CHECKPOINT = os.path.join("trained_models", "fold_{}".format(FOLD_NUM))


def main(opt):
    fold_num = int(opt.fold_num)
    epochs = int(opt.epochs)
    batch_size = int(opt.batch_size)
    learning_rate = float(opt.learning_rate)

    path_to_log = os.path.join("logs", "adv_fold_{}_{}".format(str(fold_num), str(time.time())))
    os.makedirs(path_to_log, exist_ok=True)

    path_to_metrics = os.path.join(path_to_log, "metrics.csv")
    path_to_metrics_adv = os.path.join(path_to_log, "metrics_adv.csv")

    model = ModelAdvConfFC4()

    if RELOAD_CHECKPOINT:
        print('\n Reloading checkpoint - pretrained model stored at: {} \n'.format(PATH_TO_CHECKPOINT))
        model.load(PATH_TO_CHECKPOINT)

    model.print_network()
    model.log_network(path_to_log)
    model.set_optimizer(learning_rate)

    training_set = ColorCheckerDataset(train=True, folds_num=fold_num)
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=20, drop_last=True)
    print("\n Training set size ... : {}".format(len(training_set)))

    test_set = ColorCheckerDataset(train=False, folds_num=fold_num)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=20, drop_last=True)
    print(" Test set size ....... : {}\n".format(len(test_set)))

    path_to_vis = os.path.join(path_to_log, "test_vis")
    if TEST_VIS_IMG:
        print("Test vis for monitored image {} will be saved at {}\n".format(TEST_VIS_IMG, path_to_vis))
        os.makedirs(path_to_vis)

    print("\n**************************************************************")
    print("\t\t\t Training FC4 - Fold {}".format(fold_num))
    print("**************************************************************\n")

    evaluator = Evaluator()
    evaluator_adv = Evaluator()
    best_val_loss, best_metrics = 100.0, evaluator.get_best_metrics()
    best_val_loss_adv, best_metrics_adv = 100.0, evaluator.get_best_metrics()
    train_loss, val_loss = LossTracker(), LossTracker()
    train_loss_adv, val_loss_adv = LossTracker(), LossTracker()

    for epoch in range(epochs):

        model.train_mode()
        train_loss.reset()
        train_loss_adv.reset()
        start = time.time()

        for i, (img, label, _) in enumerate(training_loader):
            img, label = img.to(DEVICE), label.to(DEVICE)
            loss, loss_adv = model.optimize(img, label)
            train_loss.update(loss)
            train_loss_adv.update(loss_adv)

            if i % 5 == 0:
                print("[ Epoch: {}/{} - Batch: {} ] | [ Train loss: {:.4f} - Train loss adv: {:.4f} ]"
                      .format(epoch, epochs, i, loss, loss_adv))
                break

        train_time = time.time() - start

        val_loss.reset()
        val_loss_adv.reset()
        start = time.time()

        if epoch % 5 == 0:
            evaluator.reset_errors()
            evaluator_adv.reset_errors()
            model.evaluation_mode()

            print("\n--------------------------------------------------------------")
            print("\t\t\t Validation")
            print("--------------------------------------------------------------\n")

            with torch.no_grad():
                for i, (img, label, file_name) in enumerate(test_loader):
                    img, label = img.to(DEVICE), label.to(DEVICE)
                    img_id = file_name[0].split(".")[0]

                    (pred, rgb, confidence), (pred_adv, rgb_adv, confidence_adv) = model.predict(img)
                    loss = model.get_angular_loss(pred, label).item()
                    loss_adv = model.get_angular_loss(pred_adv, label).item()

                    if img_id in TEST_VIS_IMG:
                        model.save_vis({"img": img, "label": label, "pred": pred, "rgb": rgb, "c": confidence},
                                       os.path.join(path_to_vis, img_id, "epoch_{}.png".format(epoch)))

                    val_loss.update(loss)
                    val_loss_adv.update(loss_adv)

                    evaluator.add_error(model.get_angular_loss(pred, label).item())
                    evaluator_adv.add_error(model.get_angular_loss(pred_adv, label).item())

                    if i % 5 == 0:
                        print("[ Epoch: {}/{} - Batch: {}] | Val loss: {:.4f} - Val loss adv: {:.4f}]"
                              .format(epoch, epochs, i, loss, loss_adv))
                        break

            print("\n--------------------------------------------------------------\n")

        val_time = time.time() - start

        metrics = evaluator.compute_metrics()
        metrics_adv = evaluator_adv.compute_metrics()
        print("\n********************************************************************")
        print(" Train Time ....... : {:.4f}".format(train_time))
        print(" Train Loss ....... : {:.4f}".format(train_loss.avg))
        print(" Train Loss Adv ... : {:.4f}".format(train_loss_adv.avg))
        if val_time > 0.1:
            print("....................................................................")
            print(" Val Time ......... : {:.4f}".format(val_time))
            print(" Val Loss ......... : {:.4f}".format(val_loss.avg))
            print(" Val Loss Adv ..... : {:.4f}".format(val_loss.avg))
            print("....................................................................")
            print_metrics(metrics, best_metrics)
            print("....................................................................")
            print("\n Adversary metrics \n")
            print_metrics(metrics_adv, best_metrics_adv)
        print("********************************************************************\n")

        if 0 < val_loss.avg < best_val_loss:
            best_val_loss = val_loss.avg
            best_metrics = evaluator.update_best_metrics()
            print("Saving new best models... \n")
            model.save(path_to_log)

        log_metrics(train_loss.avg, val_loss.avg, metrics, best_metrics, path_to_metrics)
        log_metrics(train_loss_adv.avg, val_loss_adv.avg, metrics_adv, best_metrics_adv, path_to_metrics_adv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_num", type=str, default=FOLD_NUM)
    parser.add_argument("--epochs", type=str, default=EPOCHS)
    parser.add_argument('--batch_size', type=str, default=BATCH_SIZE)
    parser.add_argument('--learning_rate', type=str, default=LEARNING_RATE)
    opt = parser.parse_args()
    main(opt)
