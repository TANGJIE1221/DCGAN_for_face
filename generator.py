import torch
from torch import nn
from config import HP


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.projection_layer = nn.Linear(HP.z_dim, 4*4*1024) # 1. feature/data transform 2. shape transform

        self.generator = nn.Sequential(

            # TransposeConv layer: 1
            nn.ConvTranspose2d(in_channels=1024,    # [N, 512, 8, 8]
                               out_channels=512,
                               kernel_size=(4, 4),
                               stride=(2, 2),
                               padding=(1, 1),
                               bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # TransposeConv layer: 2
            nn.ConvTranspose2d(in_channels=512,  # [N, 256, 16, 16]
                               out_channels=256,
                               kernel_size=(4, 4),
                               stride=(2, 2),
                               padding=(1, 1),
                               bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # TransposeConv layer: 3
            nn.ConvTranspose2d(in_channels=256,  # [N, 128, 32, 32]
                               out_channels=128,
                               kernel_size=(4, 4),
                               stride=(2, 2),
                               padding=(1, 1),
                               bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # TransposeConv layer: final
            nn.ConvTranspose2d(in_channels=128,  # [N, 3, 64, 64]
                               out_channels=HP.data_channels,   # output channel: 3 (RGB)
                               kernel_size=(4, 4),
                               stride=(2, 2),
                               padding=(1, 1),
                               bias=False),

            nn.Tanh()  # [0, 1] Relu [0, inf]
        )

    def forward(self, latent_z):    # latent space (Ramdon Input / Noise) : [N, 100]
        z = self.projection_layer(latent_z) # [N, 4*4*1024]
        z_projected = z.view(-1, 1024, 4, 4) # [N, 1024, 4, 4]: NCHW
        return self.generator(z_projected)

    @staticmethod
    def weights_init(layer):
        layer_class_name = layer.__class__.__name__
        if 'Conv' in layer_class_name:
            nn.init.normal_(layer.weight.data, 0.0, 0.02)
        elif 'BatchNorm' in layer_class_name:
            nn.init.normal_(layer.weight.data, 1.0, 0.02)
            nn.init.normal_(layer.bias.data, 0.)


if __name__ == '__main__':
    z = torch.randn(size=(64, 100))
    G = Generator()
    g_out = G(z)    # generator output
    print(g_out.size())

    import matplotlib.pyplot as plt
    import torchvision.utils as vutils
    from dataset_face import invTrans

    # format into 8x8 image grid
    grid = vutils.make_grid(g_out, nrow=8)  #
    plt.imshow(invTrans(grid).permute(1, 2, 0))  # NHWC
    plt.show()