"""
PAHdb XML Parser
Cameron Jones, 2019
"""

import click
import xmltodict
import numpy as np
import json

default_db = 'pahdb/pahdb-theoretical-3.0.xml'

@click.command()
@click.option('--input', '-i', default=default_db, help='PAHdb XML input filename.')
@click.option('--output', '-o', default=None, help='Parsed PAHdb output filename.')
@click.option('--blacklist', '-b', multiple=True, help='Blacklisted UIDs.')
def parse_db(input, output, blacklist):
    molecule_data = []
    print('Parsing XML database.')
    with open(input) as data:
        db = xmltodict.parse(data.read())
    db_type = db['pahdatabase']['@database']
    species = db['pahdatabase']['species']['specie']
    out = output or 'pahdb/pahdb-' + db_type + '.json'
    print('Input file:', input)
    print('Output file:', out)
    print('Blacklisted UIDs: [', ', '.join(blacklist) if len(blacklist) > 0 else 'none', ']')
    print('Transforming to JSON.')
    for specie in species:
        uid = int(specie['@uid'])
        if uid in blacklist:
            continue
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
            molecule['transitions'].append([wavenumber, intensity])
        molecule_data.append(molecule)
    get_bounds = lambda part: [transition[part] for molecule in molecule_data for transition in molecule['transitions']]
    json_string = json.dumps({
        'version': db['pahdatabase']['@version'],
        'full': db['pahdatabase']['@full'],
        'stats': {
            'wavenumber_max': max(get_bounds(0)),
            'wavenumber_min': min(get_bounds(0)),
            'intensity_max': max(get_bounds(1)),
            'intensity_min': min(get_bounds(1)),
        },
        'uids': [molecule['uid'] for molecule in molecule_data],
        'data': molecule_data
    }, separators=(',', ':'))
    output = open(out, 'w')
    output.write(json_string)
    print('Outputted', str(len(molecule_data)), 'molecules.')

if __name__ == '__main__':
    parse_db()
