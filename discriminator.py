# only face images, no target / label
from config import HP
from torchvision import transforms as T # torchaudio(speech) / torchtext(text)
import torchvision.datasets as TD
from torch.utils.data import DataLoader
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # openKMP cause unexpected error

# apply a label to corresponding
data_face = TD.ImageFolder(root=HP.data_root,
                           transform=T.Compose([
                               T.Resize(HP.image_size), # 64x64x3
                               T.CenterCrop(HP.image_size),
                               T.ToTensor(),    # to [0, 1]
                               T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    # can't apply ImageNet statistic
                           ]),
                           )

face_loader = DataLoader(data_face,
                         batch_size=HP.batch_size,
                         shuffle=True,
                         num_workers=HP.n_workers) # 2 workers

# normalize: x_norm = (x - x_avg) / std de-normalize: x_denorm = (x_norm * std) + x_avg
invTrans = T.Compose([
    T.Normalize(mean=[0., 0., 0.], std=[1/0.5, 1/0.5, 1/0.5]),
    T.Normalize(mean=[-0.5, -0.5, -0.5], std=[1., 1., 1.]),
])

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils

    for data, _ in face_loader:
        print(data.size()) # NCHW
        # format into 8x8 image grid
        grid = vutils.make_grid(data, nrow=8) #
        plt.imshow(invTrans(grid).permute(1, 2, 0))   # NHWC
        plt.show()
        break