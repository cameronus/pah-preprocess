"""
PAHdb Dataset Generator
Cameron Jones, 2019
"""

import click
import json
import numpy as np
import scipy as scipy
import datetime

DEFAULT_DB = 'pahdb/pahdb-theoretical.json'
NUM_SPECIES = 1000
MIX_SIZE = 10
NUM_TRAINING = 10000
NUM_TESTING = 1000
WAVE_SIGMA = 7.5
INT_SIGMA = 0.2
FWHM = 63.69 # 15cm^-1
POI = ()

@click.command()
@click.option('--input', '-i', default=DEFAULT_DB, help='PAHdb JSON input filename.')
@click.option('--num_species', default=NUM_SPECIES, help='The number of PAH species used out of the total to create the datasets.')
@click.option('--mix_size', default=MIX_SIZE, help='The number of PAH species in each mixture.')
@click.option('--num_training', default=NUM_TRAINING, help='Number of training samples.')
@click.option('--num_testing', default=NUM_TESTING, help='Number of testing samples.')
@click.option('--wave_sigma', default=WAVE_SIGMA, help='Standard deviation for wavenumber noise.')
@click.option('--int_sigma', default=INT_SIGMA, help='Standard deviation for intensity noise.')
@click.option('--fwhm', default=FWHM, help='Full width half maximum for convolution.')
@click.option('--poi', default=POI, type=(float, float), multiple=True, help='Points of interest and their width in cm^-1.')
def generate_dataset(input, num_species, mix_size, num_training, num_testing, wave_sigma, int_sigma, fwhm, poi):
    now = datetime.datetime.now().strftime('%Y-%m-%d_%I-%M%p')
    filename = f'training_n{num_species}-m{mix_size}-p{len(poi)}_{now}.npy'
    print('Input file:', input)
    print('Points of Interest: [', ', '.join(map(str, poi)), ']')
    print('Importing PAHdb.')
    print(now)
    print(filename)

    with open(input) as file:
        data = json.loads(file.read())
    # print(data)

if __name__ == '__main__':
    generate_dataset()

"""
mix_size: 2
num species: 100
poi: (200, 10) (1000, 10) (800, 40)
5 training
1 testing

outputted files:
training_n100-m2-p3_2019-03-09_11-21PM.npy =>
[
    [
        [ poi 1 from molecule 1], # all poi 1 are same length
        []
    ],
    [
        [ poi 2 from molecule 2 ] # all poi 2 are same length
    ],
 []  # poi 3
]

testing_n100-m2-p3_2019-03-09_11-21PM.npy =>


"""
