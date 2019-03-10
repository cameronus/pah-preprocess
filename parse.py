"""
PAHdb XML Parser
Cameron Jones, 2019
"""

import click
import xmltodict
import numpy as np
import json
import os

DEFAULT_DB = 'pahdb-theoretical-3.0.xml'
CUTOFF = 1000.0

@click.command()
@click.option('--input', '-i', default=DEFAULT_DB, help='PAHdb XML input filename.')
@click.option('--output', '-o', default=None, help='Parsed PAHdb output filename.')
@click.option('--blacklist', '-b', multiple=True, help='Blacklisted UIDs.')
@click.option('--cutoff', default=CUTOFF, help='Max intensity value to be allowed in data.')
def parse_db(input, output, blacklist, cutoff):
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
    cut = 0
    for specie in species:
        uid = int(specie['@uid'])
        if uid in blacklist:
            continue
        molecule = {
            'uid': uid,
            'transitions': []
        }
        transitions = specie['transitions']['mode']
        acceptable = True
        for transition in transitions:
            intensity = np.float(transition['intensity'])
            if intensity == 0.0:
                continue
            if intensity > cutoff:
                cut += 1
                acceptable = False
            wavenumber = np.float(transition['frequency']) if db_type == 'experimental' else np.float(transition['frequency']['#text'])
            molecule['transitions'].append([wavenumber, intensity, 1.0 if db_type == 'experimental' else np.float(transition['frequency']['@scale'])])
        if acceptable:
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
    print(cut, 'cut from data (intensity > %f).' % cutoff)
    print('Outputted', str(len(molecule_data)), 'molecules from', len(species), 'initial.')

if __name__ == '__main__':
    parse_db()
