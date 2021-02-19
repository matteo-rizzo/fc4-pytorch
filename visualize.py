import os
import time

import matplotlib.pyplot as plt
import torch.utils.data
from torchvision.transforms import transforms

from auxiliary.settings import DEVICE
from auxiliary.utils import correct
from classes.data.ColorCheckerDataset import ColorCheckerDataset
from classes.fc4.ModelFC4 import ModelFC4
from classes.training.Evaluator import Evaluator

NUM_SAMPLES = 1
NUM_FOLDS = 1
PATH_TO_SAVED = os.path.join("results", "fc4_cwp_{}".format(time.time()))


def main():
    evaluator = Evaluator()
    model = ModelFC4()
    os.makedirs(PATH_TO_SAVED)

    for num_fold in range(NUM_FOLDS):
        test_set = ColorCheckerDataset(train=False, folds_num=num_fold)
        dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=20)
        print('\n Length of fold {}: {} \n'.format(num_fold, len(test_set)))

        # model.load(path_to_pretrained=os.path.join("trained_models", "fold{}.pth".format(num_fold)))
        model.load(path_to_pretrained=os.path.join("trained_models", "fc4_cwp", "model.pth"))
        model.evaluation_mode()

        with torch.no_grad():
            for i, data in enumerate(dataloader):
                if NUM_SAMPLES > -1 and i > NUM_SAMPLES - 1:
                    break

                img, label, file_name = data
                img, label = img.to(DEVICE), label.to(DEVICE)
                pred, rgb, confidence = model.predict(img, return_steps=True)
                loss = model.get_angular_loss(pred, label)
                evaluator.add_error(loss.item())
                file_name = file_name[0]
                print('\t - Input: {} - Batch: {} | Loss: {:f}'.format(file_name, i, loss.item()))

                original = transforms.ToPILImage()(img.squeeze()).convert("RGB")
                gt_corrected = correct(original, label)
                est_corrected = correct(original, pred)

                fig, axs = plt.subplots(2, 3)

                axs[0, 0].imshow(original)
                axs[0, 0].set_title("Original")
                axs[0, 0].axis("off")

                axs[0, 1].imshow(confidence.squeeze())
                axs[0, 1].set_title("Confidence")
                axs[0, 1].axis("off")

                axs[0, 2].imshow(rgb.squeeze().permute(1, 2, 0))
                axs[0, 2].set_title("Estimate")
                axs[0, 2].axis("off")

                p = (rgb * confidence).squeeze()
                p -= p.min()
                p /= p.max()
                axs[1, 0].imshow(p.permute(1, 2, 0))
                axs[1, 0].set_title("Weighted Estimate")
                axs[1, 0].axis("off")

                axs[1, 1].imshow(est_corrected)
                axs[1, 1].set_title("Prediction")
                axs[1, 1].axis("off")

                axs[1, 2].imshow(gt_corrected)
                axs[1, 2].set_title("Ground Truth")
                axs[1, 2].axis("off")

                fig.tight_layout(pad=0.25)

                path_to_save = os.path.join(PATH_TO_SAVED, "fold_{}".format(num_fold), file_name)
                os.makedirs(path_to_save)

                fig.savefig(os.path.join(path_to_save, "stages.png"), bbox_inches='tight', dpi=200)
                original.save(os.path.join(path_to_save, "original.png"))
                est_corrected.save(os.path.join(path_to_save, "est_corrected.png"))
                gt_corrected.save(os.path.join(path_to_save, "gt_corrected.png"))

    metrics = evaluator.compute_metrics()
    print("\n Mean ............ : {}".format(metrics["mean"]))
    print(" Median .......... : {}".format(metrics["median"]))
    print(" Trimean ......... : {}".format(metrics["trimean"]))
    print(" Best 25% ........ : {}".format(metrics["bst25"]))
    print(" Worst 25% ....... : {}".format(metrics["wst25"]))
    print(" Percentile 95 ... : {} \n".format(metrics["wst5"]))


if __name__ == '__main__':
    main()
