import skimage.io
import skimage.transform
import numpy as np
import glob

INPUT_SCALE = 64
IMAGES_PATH = glob.glob("../../ANNOTATION_TOOL/SANOFI_ITALY_PIERRE_catalog/*/*.jpg")

def get_training_data(images_path):
    train_data = []
    for image_path in images_path:
        try:
            im = skimage.io.imread(image_path)
            im = skimage.transform.resize(im, output_shape=[INPUT_SCALE, INPUT_SCALE])
            train_data.append(im)
        except IOError:
            print("cannot rescale ", image_path)
    train_data = np.array(train_data, dtype=np.float32)
    return train_data


if __name__ == "__main__":
    train_data = get_training_data(IMAGES_PATH)
    print train_data.shape
    np.save('training_data.npy', train_data)
