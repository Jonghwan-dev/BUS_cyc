import numpy as np
import cv2
import torch
from scipy.signal import fftconvolve
from skimage import metrics

                   
    
def PSNR(self, fake, real):
    x, y = np.where(real != -1)  # Exclude background
    mse = np.mean(((fake[x,y]+1)/2. - (real[x,y]+1)/2.) ** 2)
    if mse < 1.0e-10:
        return 100
    else:
        PIXEL_MAX = 1
        return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
            
def MAE(self, fake, real):
    x, y = np.where(real != -1)  # Exclude background
    mae = np.abs(fake[x,y]-real[x,y]).mean()
    return mae/2  # from (-1,1) normalize to (0,1)
    
def save_deformation(self, defms, root):
    heatmapshow = None
    defms_ = defms.data.cpu().float().numpy()
    dir_x = defms_[0]
    dir_y = defms_[1]
    x_max, x_min = dir_x.max(), dir_x.min()
    y_max, y_min = dir_y.max(), dir_y.min()
    dir_x = ((dir_x-x_min)/(x_max-x_min))*255
    dir_y = ((dir_y-y_min)/(y_max-y_min))*255
    tans_x = cv2.normalize(dir_x, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    tans_x = cv2.applyColorMap(tans_x, cv2.COLORMAP_JET)
    tans_y = cv2.normalize(dir_y, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    tans_y = cv2.applyColorMap(tans_y, cv2.COLORMAP_JET)
    gradxy = cv2.addWeighted(tans_x, 0.5, tans_y, 0.5, 0)
    cv2.imwrite(root, gradxy)

def normxcorr2(self, template, image, mode="full"):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs. 
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the 'full' output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)
    
    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    with np.errstate(divide='ignore', invalid='ignore'): 
        out = out / np.sqrt(image * template)

    out[np.where(np.logical_not(np.isfinite(out)))] = 0
    
    return out