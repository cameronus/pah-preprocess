"""
PAHdb Dataset Generator
Cameron Jones, 2019
"""

import click
import json
import numpy as np
import scipy as scipy

DEFAULT_DB = 'pahdb/pahdb-theoretical.json'
NUM_SPECIES = 1000
MIX_SIZE = 10
NUM_TRAINING = 10000
NUM_TESTING = 1000
WAVE_SIGMA = 7.5
INT_SIGMA = 0.2

@click.command()
@click.option('--input', '-i', default=DEFAULT_DB, help='PAHdb JSON input filename.')
@click.option('--num_species', default=NUM_SPECIES, help='The number of PAH species used out of the total to create the datasets.')
@click.option('--mix_size', default=MIX_SIZE, help='The number of PAH species in each mixture.')
@click.option('--num_training', default=NUM_TRAINING, help='Number of training samples.')
@click.option('--num_testing', default=NUM_TESTING, help='Number of testing samples.')
@click.option('--wave_sigma', default=WAVE_SIGMA, help='Standard deviation for wavenumber noise.')
@click.option('--int_sigma', default=INT_SIGMA, help='Standard deviation for intensity noise.')
def generate_dataset(input, num_species, mix_size, num_training, num_testing):
    print('Input file:', input)
    print('Importing dataset.')
    with open(input) as file:
        data = json.loads(file.read())
    print(data)

if __name__ == '__main__':
    generate_dataset()
