import os
from typing import Union

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL.Image import Image

from auxiliary.settings import DEVICE


def log_metrics(train_loss: float, val_loss: float, current_metrics: dict, best_metrics: dict, path_to_log: str):
    log_data = pd.DataFrame({
        "train_loss": [train_loss],
        "val_loss": [val_loss],
        "best_mean": best_metrics["mean"],
        "best_median": best_metrics["median"],
        "best_trimean": best_metrics["trimean"],
        "best_bst25": best_metrics["bst25"],
        "best_wst25": best_metrics["wst25"],
        "best_wst5": best_metrics["wst5"],
        **{k: [v] for k, v in current_metrics.items()}
    })
    header = log_data.keys() if not os.path.exists(path_to_log) else False
    log_data.to_csv(path_to_log, mode='a', header=header, index=False)


def print_metrics(current_metrics: dict, best_metrics: dict):
    print(" Mean ......... : {:.4f} (Best: {:.4f})".format(current_metrics["mean"], best_metrics["mean"]))
    print(" Median ....... : {:.4f} (Best: {:.4f})".format(current_metrics["median"], best_metrics["median"]))
    print(" Trimean ...... : {:.4f} (Best: {:.4f})".format(current_metrics["trimean"], best_metrics["trimean"]))
    print(" Best 25% ..... : {:.4f} (Best: {:.4f})".format(current_metrics["bst25"], best_metrics["bst25"]))
    print(" Worst 25% .... : {:.4f} (Best: {:.4f})".format(current_metrics["wst25"], best_metrics["wst25"]))
    print(" Worst 5% ..... : {:.4f} (Best: {:.4f})".format(current_metrics["wst5"], best_metrics["wst5"]))


def correct(img: Image, illuminant: torch.Tensor) -> Image:
    """
    Corrects the color of the illuminant of a linear image based on an estimated (linear) illuminant
    @param img: a linear image
    @param illuminant: a linear illuminant
    @return: a non-linear color-corrected version of the input image
    """
    img = TF.to_tensor(img)

    # Correct the image
    correction = illuminant.unsqueeze(2).unsqueeze(3) * torch.sqrt(torch.Tensor([3])).to(DEVICE)
    corrected_img = torch.div(img, correction + 1e-10)

    # Normalize the image
    max_img = torch.max(torch.max(torch.max(corrected_img, dim=1)[0], dim=1)[0], dim=1)[0] + 1e-10
    max_img = max_img.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    normalized_img = torch.div(corrected_img, max_img)

    return TF.to_pil_image(linear_to_nonlinear(normalized_img).squeeze(), mode="RGB")


def linear_to_nonlinear(img: Union[np.array, Image, torch.Tensor]) -> Union[np.array, Image, torch.Tensor]:
    if isinstance(img, np.ndarray):
        return np.power(img, (1.0 / 2.2))
    if isinstance(img, torch.Tensor):
        return torch.pow(img, 1.0 / 2.2)
    return TF.to_pil_image(torch.pow(TF.to_tensor(img), 1.0 / 2.2).squeeze(), mode="RGB")


def normalize(img: np.array) -> np.array:
    return np.clip(img, 0.0, 65535.0) * (1.0 / 65535)


def rgb_to_bgr(x: np.array) -> np.array:
    return x[::-1]


def bgr_to_rgb(x: np.array) -> np.array:
    return x[:, :, ::-1]


def hwc_to_chw(x: np.array) -> np.array:
    return x.transpose(2, 0, 1)
