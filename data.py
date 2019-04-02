"""
PAHdb Dataset Generator
Cameron Jones, 2019
"""

import click
import json
import os
import numpy as np
import scipy.signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
import random
import math

mpl.use('Qt5Agg')

VECTOR_SIZE = 6000 # All spectra should fit between 0 and VECTOR_SIZE wavenumbers

DEFAULT_DB = 'pahdb/pahdb-theoretical.json' # Database to pull molecules from
CUTOFF = 100.0 # Max allowable intensity
BLACKLIST = () # Blacklisted UIDs
NUM_SPECIES = 1000 # Use the first NUM_SPECIES molecules to generate the dataset
MIX_SIZE = 10 # Number of molecules to include in each synthetic mixture
NUM_TRAINING = 800 # Number of training samples
NUM_TESTING = 200 # Number of testing samples
WAVE_SIGMA = 7.5 # Standard deviation of wavenumber noise
INT_SIGMA = 0.2 # Standard deviation of intensity noise
FWHM = 63.69 # 15cm^-1
RESOLUTION = 0.1
BUFFER = 0
POI = () # Points of interest

@click.command()
@click.option('--input', '-i', default=DEFAULT_DB, help='PAHdb JSON input filename.')
@click.option('--cutoff', default=CUTOFF, help='Max intensity value to be allowed in data.')
@click.option('--blacklist', '-b', default=BLACKLIST, type=int, multiple=True, help='Blacklisted UIDs.')
@click.option('--num_species', default=NUM_SPECIES, help='The number of PAH species used out of the total to create the datasets.')
@click.option('--mix_size', default=MIX_SIZE, help='The number of PAH species in each mixture.')
@click.option('--num_training', default=NUM_TRAINING, help='Number of training samples.')
@click.option('--num_testing', default=NUM_TESTING, help='Number of testing samples.')
@click.option('--wave_sigma', default=WAVE_SIGMA, help='Standard deviation for wavenumber noise.')
@click.option('--int_sigma', default=INT_SIGMA, help='Standard deviation for intensity noise.')
@click.option('--fwhm', default=FWHM, help='Full width half maximum for convolution.')
@click.option('--resolution', default=RESOLUTION, help='Resolution of wavenumber values.')
@click.option('--buffer', default=BUFFER, help='0-padded buffer betweeen POIs.')
@click.option('--poi', '-p', default=POI, type=(int, int), multiple=True, help='Points of interest and their width in cm^-1.')
@click.option('--no_scale', is_flag=True, help='Do not apply the scaling factor to all wavenumbers.')
def generate_dataset(input, cutoff, blacklist, num_species, mix_size, num_training, num_testing, wave_sigma, int_sigma, fwhm, resolution, buffer, poi, no_scale):
    # Get date, filename, and resolution multiplier
    now = datetime.datetime.now().strftime('%Y-%m-%d_%I-%M%p')
    filename = f'_n{num_species}-m{mix_size}-p{len(poi)}_{now}'
    res_multipler = int(1/resolution) if resolution < 1 else 1

    # Print out dataset configuration
    print('Input file:', input)
    print('Intensity Cutoff:', cutoff)
    print('Blacklisted UIDs: [', ', '.join(map(str, blacklist)) if len(blacklist) > 0 else 'none', ']')
    print('Points of Interest: [', ', '.join(map(str, poi)) if len(poi) > 0 else 'none', ']')
    if len(poi) == 0:
        print('No POIs. Exiting.')
        return
    print('Importing PAHdb.')

    # Open and parse PAHdb JSON file
    with open(input) as file:
        db = json.loads(file.read())
    uids = db['uids']
    data = db['data']
    print(len(data), 'initial molecules.')

    # Filter out all blacklisted UIDs
    data = list(filter(lambda x: x['uid'] not in blacklist, data))
    print(len(data), 'molecules after blacklisting.')

    # Filter out molecules with intensities above the cutoff
    data = list(filter(lambda molecule: all([transition[1] <= cutoff for transition in molecule['transitions']]), data))
    print(len(data), 'molecules after filtering.')

    # Scale wavenumber, and hide scaling factor
    for index, molecule in enumerate(data):
        for transition in molecule['transitions']:
            if not no_scale:
                transition[0] *= transition[2]
                del transition[2]

    # Get first num_species molecules from data
    data = data[:num_species]
    print(len(data), 'molecules after slicing.')

    # Calculate bounds
    get_bounds = lambda part: [transition[part] for molecule in data for transition in molecule['transitions']]
    stats = {
        'wavenumber_max': max(get_bounds(0)),
        'wavenumber_min': min(get_bounds(0)),
        'intensity_max': max(get_bounds(1)),
        'intensity_min': min(get_bounds(1)),
    }

    print(f"Wavenumber (min-max): {round(stats['wavenumber_min'], 2)}-{round(stats['wavenumber_max'], 2)}")
    print(f"Intensity (min-max): {round(stats['intensity_min'], 2)}-{round(stats['intensity_max'], 2)}")

    # Set vector_size
    vector_size = VECTOR_SIZE * res_multipler

    # Size of analyzed regions
    poi_length = sum([region[1] * 2 for region in poi]) * res_multipler
    buffer *= res_multipler
    data_length = poi_length + buffer * (len(poi) - 1)

    print('Length of data vector:', data_length)

    # List out POIs
    for p_index, point in enumerate(poi):
        start = max(0, (point[0] - point[1]) * res_multipler)
        stop = min(vector_size - 1, (point[0] + point[1]) * res_multipler)
        print(f'POI #{p_index + 1}: {start / res_multipler}-{stop / res_multipler} ({(stop - start) / res_multipler})')

    # Training/testing data matrices
    training_x = np.zeros((num_training, data_length), dtype=np.float32)
    training_y = np.zeros((num_training, len(uids)), dtype=np.float32)
    testing_x = np.zeros((num_testing, data_length), dtype=np.float32)
    testing_y = np.zeros((num_testing, len(uids)), dtype=np.float32)

    # Loop through to generate each sample
    for i in range(num_training + num_testing):
        index = i if i < num_training else i - num_training

        # Pick random spectra
        species = np.random.choice(data, mix_size, replace=False)
        spectrum = np.zeros((vector_size), dtype=np.float32)

        dataset = np.zeros((data_length), dtype=np.float32)
        labels = np.zeros((len(uids)), dtype=np.float32)

        if i < num_training:
            print(f'Training Sample #{i + 1}')
        else:
            print(f'Testing Sample #{i - num_training + 1}')
        print('UIDs in sample: [', ', '.join(map(str, [m['uid'] for m in species])), ']')
        print()

        # Add noise, normalize intensity, and generate labels
        for molecule in species:
            uid = molecule['uid']
            for transition in molecule['transitions']:
                wavenumber = int(round(np.random.normal(0, wave_sigma) + transition[0], 1) * res_multipler)
                intensity = (np.random.normal(0, int_sigma) + 1) * transition[1]
                # spectrum[wavenumber] += intensity / stats['intensity_max']
                spectrum[wavenumber] += intensity
            labels[uids.index(uid)] = 1.0

        # plt.plot(spectrum)
        # plt.show()

        # kernel = scipy.signal.gaussian(1001, std=fwhm)
        # spectrum = scipy.signal.convolve(spectrum, kernel, mode='same')

        # Perform a Gaussian convolution on the whole spectrum
        spectrum = scipy.ndimage.filters.gaussian_filter1d(spectrum, fwhm).astype(np.float16)

        # plt.plot(spectrum)
        # plt.show()

        # Cut spectra into POIs and rejoin with optional buffers
        count = 0
        for p_index, point in enumerate(poi):
            start = max(0, (point[0] - point[1]) * res_multipler)
            stop = min(vector_size - 1, (point[0] + point[1]) * res_multipler)
            section = spectrum[start:stop]
            for s_index, s in enumerate(section):
                dataset[count] = s
                count += 1
                if (s_index == len(section) - 1 and p_index != len(poi) - 1):
                    count += buffer

        # normalize again?

        # plt.plot(dataset)
        # plt.show()
        # return

        # Insert final spectrum and labels into dataset matrix
        if i < num_training:
            training_x[index] = dataset
            training_y[index] = labels
        else:
            testing_x[index] = dataset
            testing_y[index] = labels

    # Create dataset directory if it doesn't exist
    if not os.path.exists('datasets'):
        os.makedirs('datasets')

    # Save training and testing data
    np.savez('datasets/training' + filename, x=training_x, y=training_y)
    np.savez('datasets/testing' + filename, x=testing_x, y=testing_y)

    print('Dataset saved successfully.')

if __name__ == '__main__':
    generate_dataset()
