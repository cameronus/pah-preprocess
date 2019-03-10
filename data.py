"""
PAHdb Dataset Creator
Cameron Jones, 2019
"""

import click
import json
import numpy as np
import scipy as scipy

default_db = 'pahdb/pahdb-theoretical.json'

@click.command()
@click.option('--input', '-i', default=default_db, help='PAHdb XML input filename.')
def generate_dataset(input):
    print('...')

if __name__ == '__main__':
    generate_dataset()
