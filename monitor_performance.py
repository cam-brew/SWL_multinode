import logging
import multiprocessing
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import animation

"""
    Visualization functions:
    - setup_logger(): for reporting parallelization task updates and performance
    - animate_stack(): plot raw tomograph, masked foreground, and label field for each slice of stack and display
    - plot_gmm_masked_clusters(): plot of greyscale histogram overlaid by GMM generated fit function for label assignment
"""

    
def setup_logger():
    logger = multiprocessing.get_logger()
    handler = logging.StreamHandler()
    # formatter = logging.Formatter('[%(levelname)s | %(message)s]')
    # handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
    
def animate_stack(tomo_stack,mask_stack,label_stack,rank):
    depth = tomo_stack.shape[0]
    
    writer = animation.PillowWriter(fps=15)
    
    fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    # Plot initial slices
    im0 = ax[0].imshow(tomo_stack[0, :, :], cmap='gray')
    ax[0].set_title('Raw Data')

    im1 = ax[1].imshow(mask_stack[0, :, :], cmap='gray')
    ax[1].set_title('Binary Mask')

    # For labeled image, determine min and max labels for color scale
    vmin = np.min(label_stack)
    vmax = np.max(label_stack)

    im2 = ax[2].imshow(label_stack[0, :, :], cmap='gray', vmin=vmin, vmax=vmax)
    ax[2].set_title('Labeled')

    # Add colorbar without shrinking the image axis
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im2, cax=cax, ticks=np.arange(vmin, vmax + 1),cmap='gray')
    cbar.ax.set_ylabel('Labels')

    # Animation update function
    def update(i):
        im0.set_data(tomo_stack[i, :, :])
        im1.set_data(mask_stack[i, :, :])
        im2.set_data(label_stack[i, :, :])
        # No need to update colorbar limits since vmin and vmax fixed
        return im0, im1, im2

    ani = animation.FuncAnimation(fig, update, frames=depth, interval=15, blit=False, repeat=True)

    plt.show()

def plot_gmm_masked_clusters(image_slice, mask_slice, label_slice, gmm_model,transparency):
    """
    Plots original image, masked GMM labels, and intensity histogram with GMM fit.
    """
    masked_pixels = image_slice[mask_slice].reshape(-1, 1)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(image_slice, cmap='gray')

    # Cluster labels (only within mask)
    n_classes = gmm_model.n_components
    display_labels = np.full_like(label_slice, np.nan, dtype=float)  # background is NaN
    display_labels[mask_slice] = label_slice[mask_slice]  # only show masked voxels
    cmap = mpl.colormaps.get_cmap('tab10').resampled(n_classes)
    # cmap = plt.cm.get_cmap('tab10', n_classes)
    im = ax[0].imshow(display_labels, cmap=cmap, vmin=0, vmax=n_classes - 1, alpha=transparency)
    cb = plt.colorbar(im, ax=ax[0], ticks=range(n_classes))
    cb.set_label("Cluster")
    ax[0].set_title("Raw overlaid with GMM Labels")

    # Histogram + GMM density fit
    ax[1].hist(masked_pixels, bins=255, density=True, alpha=0.3, color='gray', label='Masked Intensities')
    x_vals = np.linspace(masked_pixels.min(), masked_pixels.max(), 1000).reshape(-1, 1)
    logprob = gmm_model.score_samples(x_vals)
    pdf = np.exp(logprob)
    ax[1].plot(x_vals, pdf, color='red', lw=2, label='GMM Fit')
    ax[1].set_title("1D GMM on Masked Pixels")
    ax[1].legend()

    plt.tight_layout()
    plt.show()