import os
import gzip
import dill

from dps.datasets.base import ImageDataset, ImageFeature

class AtariImageFileDataset(ImageDataset):
    def __init__(self, load_path="~/dps_data/raw_image/atari"):
        super(AtariImageFileDataset).__init__(self)
        self.load_path = load_path

    @property
    def obs_shape(self):
        """ Return the shape of your images. """
        return (210, 160, 3)

    def features(self):
        if self._features is None:
            self._features = [ImageFeature("image", self.obs_shape)]
        return self._features

    def _make(self):
        # load dataset in current form
        if self.file_names == "all":
            # fetch all files in subdirectory
            self.filenames = [f for f in os.listdir(self.load_path) if os.path.isfile(os.path.join(self.load_path, f))]

        images = []
        for name in self.file_names:
            images.append(load_gameframe(name, load_path))

        for image in images:
            self._write_example(image=image)


def load_gameframe(file_name, save_dir='../data/static_gym/'):
    imgpath = os.path.join(save_dir, file_name)
    with gzip.open(imgpath, 'rb') as f:
        img = dill.load(f)
        return img