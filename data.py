"""
PAHdb Dataset Generator
Cameron Jones, 2019
"""

import click
import json
import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
import datetime
import random
import statistics as stat

DEFAULT_DB = 'pahdb/pahdb-theoretical.json'
CUTOFF = 1000.0 # Max allowable intensity
BLACKLIST = () # Blacklisted UIDs
NUM_SPECIES = 1000 # Use the first NUM_SPECIES molecules to generate the dataset
MIX_SIZE = 3 # Number of molecules to include in each synthetic mixture
NUM_TRAINING = 800 # Number of training samples
NUM_TESTING = 200 # Number of testing samples
WAVE_SIGMA = 7.5 # Standard deviation of wavenumber noise
INT_SIGMA = 0.2 # Standard deviation of intensity noise
FWHM = 63.69 # 15cm^-1
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
@click.option('--poi', '-p', default=POI, type=(float, float), multiple=True, help='Points of interest and their width in cm^-1.')
@click.option('--no_scale', is_flag=True, help='Do not apply the scaling factor to all wavenumbers.')
def generate_dataset(input, cutoff, blacklist, num_species, mix_size, num_training, num_testing, wave_sigma, int_sigma, fwhm, poi, no_scale):
    now = datetime.datetime.now().strftime('%Y-%m-%d_%I-%M%p')
    filename = f'_n{num_species}-m{mix_size}-p{len(poi)}_{now}.npy'

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
    data = db['data']

    # Filter out all blacklisted UIDs
    data = list(filter(lambda x: x['uid'] not in blacklist, data))

    # Filter out molecules with intensities above the cutoff, scale wavenumber, and hide scaling factor
    for molecule in data:
        for transition in molecule['transitions']:
            if transition[1] > cutoff:
                data.remove(molecule)
                break
            if not no_scale:
                transition[0] *= transition[2]
                del transition[2]

    # Get first num_species molecules from data
    data = data[:num_species]

    # Calculate bounds
    get_bounds = lambda part: [transition[part] for molecule in data for transition in molecule['transitions']]
    stats = {
        'wavenumber_max': max(get_bounds(0)),
        'wavenumber_min': min(get_bounds(0)),
        'intensity_max': max(get_bounds(1)),
        'intensity_min': min(get_bounds(1)),
    }
    print(stats)

    for i in range(1): # num_training + num_testing
        # print(i)
        # if i < num_training:
        #     print('train')
        # else:
        #     print('test')
        species = np.random.choice(data, mix_size, replace=False)
        species_uids = [molecule['uid'] for molecule in species]

        for molecule in species:
            # print(molecule['uid'])
            transitions = molecule['transitions']
            wavenumbers = [transition[0] for transition in transitions]
            intensities = [transition[1] for transition in transitions]
            # add noise
            wavenumbers = np.around(wavenumbers, decimals=1)
            print(wavenumbers)
            print(intensities)

            data_matrix_training = np.zeros((num_sets * num_training_molecules, VECTOR_SIZE+1), dtype=np.float32)

            # plt.hist(intensities)
            # return


        # pick mix_size randomly from num_species #
        # get UIDs #
        # if not no_scale, scale wavenumbers #
        # add noise
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

testing_n100-m2-p3_2019-03-09_11-21PM.npy =>
"""
