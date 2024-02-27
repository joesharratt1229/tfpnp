from PIL import Image
import numpy as np
import torch

from transforms import fft2_new


def fft(data):
    return torch.view_as_real(
        torch.fft.fftn(
            torch.view_as_complex(data), dim = (-2, -1), norm = 'ortho'
        )
    )

def ifft(data):
        return torch.view_as_real(
        torch.fft.ifftn(
            torch.view_as_complex(data), dim = (-2, -1), norm = 'ortho')
        )


if __name__ == '__main__':
    img = Image.open('Brain_data2.png')
    img_1 = Image.open('Brain_data1.png')
    img = np.array(img, dtype = np.float32)/255
    img_1 = torch.from_numpy(np.array(img_1, dtype = np.float32)/225)
    img = torch.from_numpy(img)
    print(img.shape)
    img_new = torch.fft.ifftshift(img, dim = (-2, -1))
    img_new = torch.fft.fftn(img_new, dim = (-2, -1), norm = 'ortho')
    img_new = torch.fft.fftshift(img_new, dim = (-2, -1))
    print(img_new)
    img_old = torch.stack([img, torch.zeros_like(img)], dim = -1)
    img_old = fft2_new(img_old)
    print(img_old)


    