"""
PAHdb XML Parser
Cameron Jones, 2019
"""

import click
import xmltodict
import numpy as np
import json
import os
import statistics as stat

DEFAULT_DB = 'pahdb-theoretical-3.0.xml'

@click.command()
@click.option('--input', '-i', default=DEFAULT_DB, help='PAHdb XML input filename.')
@click.option('--output', '-o', default=None, help='Parsed PAHdb output filename.')
def parse_db(input, output):
    molecule_data = []
    print('Parsing XML database.')
    with open(input) as data:
        db = xmltodict.parse(data.read())
    db_type = db['pahdatabase']['@database']
    species = db['pahdatabase']['species']['specie']
    out = output or 'pahdb/pahdb-' + db_type + '.json'
    print('Input file:', input)
    print('Output file:', out)
    print('Transforming to JSON.')
    for specie in species:
        uid = int(specie['@uid'])
        molecule = {
            'uid': uid,
            'transitions': []
        }
        transitions = specie['transitions']['mode']
        for transition in transitions:
            intensity = np.float(transition['intensity'])
            if intensity == 0.0:
                continue
            wavenumber = np.float(transition['frequency']) if db_type == 'experimental' else np.float(transition['frequency']['#text'])
            molecule['transitions'].append([wavenumber, intensity, 1.0 if db_type == 'experimental' else np.float(transition['frequency']['@scale'])])
        molecule_data.append(molecule)
    json_string = json.dumps({
        'version': db['pahdatabase']['@version'],
        'full': db['pahdatabase']['@full'],
        'uids': [molecule['uid'] for molecule in molecule_data],
        'data': molecule_data
    }, separators=(',', ':'))
    if not os.path.exists('pahdb') and output is None:
        os.makedirs('pahdb')
    output = open(out, 'w')
    output.write(json_string)
    print()
    print('Stats:')
    wavenumbers_pre = [transition[0] for molecule in molecule_data for transition in molecule['transitions']]
    wavenumbers = [transition[0] * transition[2] for molecule in molecule_data for transition in molecule['transitions']]
    intensities = [transition[1] for molecule in molecule_data for transition in molecule['transitions']]
    print()
    print('Pre-scale Wavenumber Max:', max(wavenumbers_pre))
    print('Pre-scale Wavenumber Min:', min(wavenumbers_pre))
    print('Scaled Wavenumber Max:', max(wavenumbers))
    print('Scaled Wavenumber Min:', min(wavenumbers))
    print()
    print('Intensity Mean:', stat.mean(intensities))
    print('Intensity Standard Deviation:', stat.stdev(intensities))
    print('Intensity Max:', max(intensities))
    print('Intensity Min:', min(intensities))
    print()
    print('Outputted', str(len(molecule_data)), 'molecules.')

if __name__ == '__main__':
    parse_db()
