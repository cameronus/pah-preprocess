"""
PAHdb Dataset Generator
Cameron Jones, 2019
"""

import click
import json
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import datetime
import random
import math

DEFAULT_DB = 'pahdb/pahdb-theoretical.json'
CUTOFF = 100.0 # Max allowable intensity
BLACKLIST = () # Blacklisted UIDs
NUM_SPECIES = 1000 # Use the first NUM_SPECIES molecules to generate the dataset
MIX_SIZE = 3 # Number of molecules to include in each synthetic mixture
NUM_TRAINING = 800 # Number of training samples
NUM_TESTING = 200 # Number of testing samples
WAVE_SIGMA = 7.5 # Standard deviation of wavenumber noise
INT_SIGMA = 0.2 # Standard deviation of intensity noise
FWHM = 63.69 # 15cm^-1
RESOLUTION = 0.1
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
@click.option('--poi', '-p', default=POI, type=(int, int), multiple=True, help='Points of interest and their width in cm^-1.')
@click.option('--no_scale', is_flag=True, help='Do not apply the scaling factor to all wavenumbers.')
def generate_dataset(input, cutoff, blacklist, num_species, mix_size, num_training, num_testing, wave_sigma, int_sigma, fwhm, resolution, poi, no_scale):
    # Get date, filename, and resolution multiplier
    now = datetime.datetime.now().strftime('%Y-%m-%d_%I-%M%p')
    filename = f'_n{num_species}-m{mix_size}-p{len(poi)}_{now}.npy'
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

    # Set vector_size
    # vector_size = 5400 * res_multipler
    vector_size = int(stats['wavenumber_max'] * res_multipler)
    print(stats)

    poi_length = sum([region[1] * 2 for region in poi]) * res_multipler
    buffer = 0
    print(poi_length)

    training_x = np.zeros((num_training, poi_length + buffer * (len(poi) - 1)), dtype=np.float32)
    training_y = np.zeros((num_training, len(uids)), dtype=np.float32)
    testing_x = np.zeros((num_testing, poi_length + buffer * (len(poi) - 1)), dtype=np.float32)
    testing_y = np.zeros((num_testing, len(uids)), dtype=np.float32)

    for i in range(num_training + num_testing):
        index = i if i < num_training else i - num_training

        species = np.random.choice(data, mix_size, replace=False)
        spectrum = np.zeros((vector_size), dtype=np.float32)
        # print(max([transition[1] for transition in molecule['transitions'] for molecule in species]))

        for molecule in species:
            uid = molecule['uid']
            for transition in molecule['transitions']:
                wavenumber = int(round(np.random.normal(0, wave_sigma) + transition[0], 1) * res_multipler)
                intensity = (np.random.normal(0, int_sigma) + 1) * transition[1]
                spectrum[wavenumber] += intensity / stats['intensity_max']
            training_y[index][uids.index(uid)] = 1.0

        print([uids[i] for i in np.where(training_y[index] == 1.0)[0]])

        plt.plot(spectrum)
        plt.show()

        # kernel = scipy.signal.gaussian(1001, std=fwhm)
        # spectrum = scipy.signal.convolve(spectrum, kernel, mode='same')

        spectrum = scipy.ndimage.filters.gaussian_filter1d(spectrum, fwhm).astype(np.float16)

        plt.plot(spectrum)
        plt.show()

        count = 0
        for point in poi:
            start = max(0, (point[0] - point[1]) * res_multipler)
            stop = min(len(spectrum) - 1, (point[0] + point[1]) * res_multipler)
            section = spectrum[start:stop]
            # print(start, 'to', stop)
            # print(len(section))
            for p_index, s in enumerate(section):
                training_x[index][count] = s
                count += 1
                # print(count)
                if (s == len(section)):
                    count += buffer

        # print(list(filter(lambda x: x > 0.0, training_x[index])))
        print(training_x[index])

            # training_x[index][count] = 0.0
            # training_x[index][count]
            # count += 1
            # if (p_index )

        # training_x[index] = scipy.ndimage.filters.gaussian_filter1d(training_x[index], fwhm).astype(np.float16)
        # print(max(spectrum))

        # kernel = scipy.signal.gaussian(1001, std=fwhm)
        # print(kernel[0])
        # training_x[index] = scipy.signal.convolve(training_x[index], kernel, mode='same')

        plt.plot(training_x[index])
        plt.show()
        return
        print(training_x[index])
        # print(list(filter(lambda x: x > 0.0, training_x[index])))
        # spectrum *= 1.0/spectrum.max()
        # print(max(spectrum))
        # print(kernel)
        # print(spectrum)

        if i < num_training:
            print('train', i)
        else:
            print('test', i - num_training)
        print()


        # pick mix_size randomly from num_species #
        # get UIDs #
        # if not no_scale, scale wavenumbers #
        # add noise #
        # merge transitions
        # convolve
        # split into POIs
        # add data to np array and save

if __name__ == '__main__':
    generate_dataset()

"""
mix_size: 2
num species: 100
poi: (200, 10) (1000, 10) (800, 40)
5 training
1 testing
(equal concentrations, boolean species detection returned from CNN)

outputted files:
training_n100-m2-p3_2019-03-09_11-21PM.npy =>
# setup 1
[
    [
        [
            [ sample 1 of poi 1 from molecule 1 ], # all poi 1 are same length
            [ uids in sample 1 ]
        ],
        [
            [ sample 2 of poi 1 from molecule 1 ],
            [ uids in sample 2 ]
        ]
    ],
    [

    ],
 []  # poi 3
]

# setup 2
[
    [
        [ sample 1 of poi 1 ], # all poi 1 are same length
        [ sample 2 of poi 1 ]
    ],
    [

    ],
    [

    ]
]

testing_n100-m2-p3_2019-03-09_11-21PM.npy =>
"""
