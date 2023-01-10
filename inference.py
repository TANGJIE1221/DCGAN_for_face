import torch
from dataset_face import face_loader, invTrans
from generator import Generator
from config import HP
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# new an generator model instance
G = Generator()
checkpoint = torch.load('./model_save/model_g_71_225000.pth', map_location='cpu')
G.load_state_dict(checkpoint['model_state_dict'])
G.to(HP.device)
G.eval() # set evaluation mode

while 1:
    # 1. Disentangled representation: manual set Z： [0.3, 0, ]
    # 2. any input: z: fuzzy image -> high resolution image / mel -> audio/speech(vocoder)

    latent_z = torch.randn(size=(HP.batch_size, HP.z_dim), device=HP.device)
    fake_faces = G(latent_z)
    grid = vutils.make_grid(fake_faces, nrow=8) # format into a "big" image
    plt.imshow(invTrans(grid).permute(1, 2, 0)) # HWC
    plt.show()
    input()