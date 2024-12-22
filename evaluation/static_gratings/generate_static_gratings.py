"""
This stimulus was used to measure the spatial frequency tuning and the orientation tuning of the cells,
providing a finer measurement of orientation than provided from the drifting grating stimulus.

Protocol from: https://observatory.brain-map.org/visualcoding/stimulus/static_gratings
"""



import numpy as np
import matplotlib.pyplot as plt


static_gratings_params = {
    'width': 64,
    'height': 64,
    'contrast': 0.8,
    'orientation': [0, 30, 60, 90, 120, 150], # degrees
    'spatial_frequency': [0.02, 0.04, 0.08, 0.16, 0.32], # cycles/degree
    'phase': [0, 0.25, 0.5, 0.75]
}





def generate_grating(width, height, contrast, orientation, spatial_frequency, phase):
    """
    Generates a full field static sinusoidal grating.

    Args:
    width (int): The width of the image in pixels.
    height (int): The height of the image in pixels.
    contrast (float): The contrast of the grating, ranging from 0 to 1.
    orientation (float): The orientation of the grating in degrees.
    spatial_frequency (float): The spatial frequency of the grating in cycles/degree.
    phase (float): The phase of the grating, where 0 is 0 degrees and 1 is 360 degrees.

    Returns:
    numpy.ndarray: The generated grating as a 2D numpy array.
    """
    # Convert orientation to radians
    orientation_rad = np.deg2rad(orientation)

    # Calculate the wavelength (the distance over which the sine wave repeats)
    wavelength = 1 / spatial_frequency

    # Convert phase to radians
    phase_rad = phase * 2 * np.pi

    # Create a coordinate system
    x = np.linspace(-width / 2, width / 2, width)
    y = np.linspace(-height / 2, height / 2, height)
    xv, yv = np.meshgrid(x, y)

    # Rotate the coordinate system by the orientation
    x_theta = xv * np.cos(orientation_rad) + yv * np.sin(orientation_rad)
    y_theta = -xv * np.sin(orientation_rad) + yv * np.cos(orientation_rad)

    # Generate the sinusoidal grating
    grating = np.sin((2 * np.pi * x_theta / wavelength) + phase_rad)

    # Apply contrast
    grating = 0.5 + contrast * grating * 0.5

    return grating


def plot_grating(grating, contrast, orientation, spatial_frequency, phase):
    """
    Plots a 2D grating.

    Args:
    grating (numpy.ndarray): The 2D grating to plot.
    """
    width, height = grating.shape
    plt.figure(figsize=(10, 7.5))
    plt.imshow(grating, cmap='gray', extent=(-width / 2, width / 2, -height / 2, height / 2))
    plt.colorbar()
    plt.title(f'Static Grating\nContrast: {contrast}, Orientation: {orientation}, Spatial Frequency: {spatial_frequency}, Phase: {phase}')
    plt.savefig(f'static_grating_c_{contrast}_o_{orientation}_sfreq_{spatial_frequency}_p_{phase}.png')
    return None

def average_luminance(image):
    """
    Computes the average luminance of a grayscale image.

    Args:
    image (numpy.ndarray): A 2D numpy array representing a grayscale image.

    Returns:
    float: The average luminance of the image.
    """
    # Compute the average of all pixel values
    avg_lum = np.mean(image)

    return avg_lum

def generate_blank_sweep(width, height, gray_level=0.5):
    """
    Generates a blank sweep of mean luminance gray.

    Args:
    width (int): The width of the image in pixels.
    height (int): The height of the image in pixels.
    gray_level (float): The gray level of the image, ranging from 0 (black) to 1 (white), default is 0.5 (mean gray).

    Returns:
    numpy.ndarray: A 2D numpy array filled with the gray level.
    """
    # Create an array filled with the gray level
    blank_sweep = np.full((height, width), gray_level)

    return blank_sweep

def plot_blank_sweep(blank_sweep, gray_level, contrast, orientation, spatial_frequency, phase):
    """
    Plots a blank sweep.

    Args:
    blank_sweep (numpy.ndarray): The 2D blank sweep to plot.
    gray_level (float): The gray level of the blank sweep.
    """
    width, height = blank_sweep.shape
    plt.figure(figsize=(10, 7.5))
    plt.imshow(blank_sweep, cmap='gray', extent=(-width / 2, width / 2, -height / 2, height / 2))
    plt.colorbar()
    plt.title(f'Blank Sweep\nGray Level: {gray_level}\nContrast: {contrast}, Orientation: {orientation}, Spatial Frequency: {spatial_frequency}, Phase: {phase}')
    plt.savefig(f'blank_sweep_c_{contrast}_o_{orientation}_sfreq_{spatial_frequency}_p_{phase}.png')
    return None


if __name__ == '__main__':

    w = static_gratings_params['width']
    h = static_gratings_params['height']
    c = static_gratings_params['contrast']
    save_dict = {}
    for o in static_gratings_params['orientation']:
        for sf in static_gratings_params['spatial_frequency']:
            for p in static_gratings_params['phase']:
                grating = generate_grating(width=w, height=h, contrast=c, orientation=o, spatial_frequency=sf, phase=p)
                plot_grating(grating, contrast=c, orientation=o, spatial_frequency=sf, phase=p)
                avg_lum = average_luminance(grating)
                blank_sweep = generate_blank_sweep(width=w, height=h)
                plot_blank_sweep(blank_sweep, avg_lum, contrast=c, orientation=o, spatial_frequency=sf, phase=p)
                
                # Save the grating and the blank sweep
                save_dict[f'grating_o_{o}_sf_{sf}_p_{p}'] = grating
                save_dict[f'blank_sweep_o_{o}_sf_{sf}_p_{p}'] = blank_sweep
                
    np.savez('static_gratings.npz', **save_dict)