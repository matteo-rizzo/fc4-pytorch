import os

import torch.utils.data

from auxiliary.settings import DEVICE
from classes.data.ColorCheckerDataset import ColorCheckerDataset
from classes.fc4.ModelFC4 import ModelFC4
from classes.training.Evaluator import Evaluator

PTH_PATHS = [os.path.join("trained_models", "fold0.pth"),
             os.path.join("trained_models", "fold1.pth"),
             os.path.join("trained_models", "fold2.pth")]


def main():
    evaluator = Evaluator()

    model = ModelFC4()

    for i in range(3):
        test_set = ColorCheckerDataset(train=False, folds_num=i)
        dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=20)
        print('\n Length of fold {}: {} \n'.format(i, len(test_set)))

        model.load(path_to_pretrained=PTH_PATHS[i])
        model.evaluation_mode()

        with torch.no_grad():
            for _, data in enumerate(dataloader):
                img, label, file_name = data
                img, label = img.to(DEVICE), label.to(DEVICE)
                pred = model.predict(img)
                loss = model.get_angular_loss(pred, label)
                evaluator.add_error(loss.item())
                print('\t - Input: {}, AE: {:f}'.format(file_name[0], loss.item()))

    metrics = evaluator.compute_metrics()
    print("\n Mean ............ : {}".format(metrics["mean"]))
    print(" Median .......... : {}".format(metrics["median"]))
    print(" Trimean ......... : {}".format(metrics["trimean"]))
    print(" Best 25% ........ : {}".format(metrics["bst25"]))
    print(" Worst 25% ....... : {}".format(metrics["wst25"]))
    print(" Percentile 95 ... : {} \n".format(metrics["pct95"]))


if __name__ == '__main__':
    main()
