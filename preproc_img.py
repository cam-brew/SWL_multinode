import os
import numpy as np
import matplotlib.pyplot as plt
import pywt
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter,median_filter,map_coordinates
from skimage.transform import warp_polar,warp,rotate
from concurrent.futures import ProcessPoolExecutor
from mpi4py import MPI

def remove_spring(local_stack, global_start):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    local_first = -1
    for i, slice2d in enumerate(local_stack):
        if 65535 in slice2d:
            local_first = global_start + i
            break

    local_first_val = local_first if local_first != -1 else np.iinfo(np.int32).max
    global_first = comm.allreduce(local_first_val,op=MPI.MIN)

    return -1 if global_first == np.iinfo(np.int32).max else global_first

def radial_ring_removal(img, radial_window=100, subtract_fraction=0.01, smooth_sigma=0.01):
    """Remove ring artifacts using radial FFT profile subtraction."""

    img = img.astype(np.float32, copy=False)

    h, w = img.shape
    center = (h // 2, w // 2)

    # FFT
    fft = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(fft).astype(np.float32,copy=False)
    phase = np.angle(fft).astype(np.float32,copy=False)

    # Radial statistics
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - center[1])**2 + (Y - center[0])**2).astype(np.int32,copy=False)
    r_flat = r.ravel()
    mag_flat = mag.ravel()

    # Bincount for radial mean
    radial_sum = np.bincount(r_flat, weights=mag_flat)
    radial_count = np.bincount(r_flat)
    radial_mean = np.divide(radial_sum, radial_count, out=np.zeros_like(radial_sum), where=radial_count != 0)

    # Smooth and correct
    radial_smooth = ndi.gaussian_filter1d(radial_mean, radial_window)
    radial_corrected = radial_mean - subtract_fraction * radial_smooth
    radial_corrected = np.clip(radial_corrected, 0, None)

    # Apply corrected radial profile
    mag_corrected = mag - radial_mean[r] + radial_corrected[r]
    fft_corrected = mag_corrected * np.exp(1j * phase)

    # Inverse FFT + smoothing
    img_filtered = np.fft.ifft2(np.fft.ifftshift(fft_corrected)).real
    img_smoothed = ndi.gaussian_filter(img_filtered.astype(np.float32,copy=False), sigma=smooth_sigma)

    # Normalize result
    return img_smoothed

def ring_remove_test(img,wavelet='db11',level=None,sigma=None):
    
    h,w = img.shape
    if level == None:
        level = min(2,pywt.dwtn_max_level((h,w),wavelet))
    if sigma == None:
        sigma = w / 100.0
        
    coeffs = pywt.wavedec2(img,wavelet=wavelet,level=level)
    cA, detail_coeffs = coeffs[0],coeffs[1:]
    new_detail_coeffs = []
    
    for (cH,cV,cD) in detail_coeffs:
        sigma_scaled = max(1, cV.shape[0] / 20.0)
        cV_filt = cV - gaussian_filter1d(cV,sigma=sigma_scaled,axis=0)
        new_detail_coeffs.append((cH,cV_filt,cD))
        
    corr_img = pywt.waverec2([cA] + new_detail_coeffs, wavelet=wavelet)
    corr_img = corr_img[:img.shape[0],:img.shape[1]]
    
    # corr_img = np.mean((corr_img,scipy.signal.medfilt2d(img,kernel_size=25)),axis=0)
    return corr_img

def gauss_2d_slices(vol,sigma):
    if np.isscalar(sigma):
        sigma= (0,sigma,sigma)
    elif len(sigma) == 2:
        sigma = (0,) + tuple(sigma)
    elif len(sigma) == 3:
        sigma = tuple(sigma)

    else:
        raise ValueError("Sigma invalid shape")
    
    return gaussian_filter(vol,sigma=sigma)
def linear_polar(img, o=None, r=None, output=None, order=1):
    if o is None:
        o = np.array(img.shape[:2]) / 2 - 0.5
    if r is None:
        r = np.hypot(*img.shape[:2]) / 2
    if output is None:
        shp = int(round(r)), int(round(r * 2 * np.pi))
        output = np.zeros(shp, dtype=img.dtype)
    elif isinstance(output, tuple):
        output = np.zeros(output, dtype=img.dtype)

    out_h, out_w = output.shape
    rs = np.linspace(0, r, out_h)
    ts = np.linspace(0, 2 * np.pi, out_w)
    xs = rs[:, None] * np.cos(ts) + o[1]
    ys = rs[:, None] * np.sin(ts) + o[0]
    map_coordinates(img, [ys, xs], order=order, output=output)
    return output

def polar_linear(polar_img, o=None, r=None, output=None, order=1):
    if r is None:
        r = polar_img.shape[0]
    if output is None:
        output = np.zeros((r * 2, r * 2), dtype=polar_img.dtype)
    elif isinstance(output, tuple):
        output = np.zeros(output, dtype=polar_img.dtype)

    if o is None:
        o = np.array(output.shape) / 2 - 0.5

    out_h, out_w = output.shape
    ys, xs = np.mgrid[:out_h, :out_w]
    ys = ys - o[0]
    xs = xs - o[1]
    rs = np.sqrt(xs**2 + ys**2)
    ts = np.arccos(xs / (rs + 1e-10))
    ts[ys < 0] = 2 * np.pi - ts[ys < 0]
    ts *= (polar_img.shape[1] - 1) / (2 * np.pi)

    map_coordinates(polar_img, [rs, ts], order=order, output=output)
    return output

def ring_remove_polar(img,sigma=6):
    center = np.array(img.shape) / 2 - 0.5
    radius = np.hypot(*img.shape) / 2
    
    polar = linear_polar(img, o=center, r=radius)

    # Apply Gaussian filtering along radius (axis=0) to remove horizontal stripes
    filtered = median_filter(polar, size=(7,2))
    
    corrected = polar_linear(filtered, o=center, r=polar.shape[0], order=3, output=img.shape)
    return corrected


def process_tomo_stack(volume_3d, num_workers=None):
    """Process a full 3D volume with multiprocessing."""
    if num_workers is None:
        num_workers = os.cpu_count()
        
    # if volume_3d.dtype != np.float32:
    #     volume_3d = volume_3d.astype(np.float32)
    
    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        results = list(exe.map(ring_remove_polar,volume_3d))
            
    return np.stack(results,axis=0)

def show_comparison(original, cleaned, slice_idx=None, vmin=None, vmax=None):
    """Show before/after comparison of a selected slice."""
    if slice_idx is None:
        slice_idx = original.shape[0] // 2
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original[slice_idx], cmap='gray', vmin=vmin, vmax=vmax)
    ax[0].set_title("Original Slice")
    ax[0].axis('off')
    ax[1].imshow(cleaned[slice_idx], cmap='gray', vmin=vmin, vmax=vmax)
    ax[1].set_title("Cleaned Slice")
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()
    
