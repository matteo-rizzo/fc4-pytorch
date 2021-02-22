import os

import imageio

PATH_TO_IMAGES = os.path.join("logs", "test_vis_IMG_0753")
GIF_NAME = "test"


def main():
    file_names = [file_name for file_name in os.listdir(PATH_TO_IMAGES) if "stages" in file_name]
    file_names.sort(key=lambda x: int(x.split("_")[0].split("epoch")[-1]))
    images = []
    for file_name in file_names:
        print(file_name)
        images += [imageio.imread(os.path.join(PATH_TO_IMAGES, file_name))]
    imageio.mimsave(os.path.join(PATH_TO_IMAGES, "{}.gif".format(GIF_NAME)), images)


if __name__ == '__main__':
    main()
