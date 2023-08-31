# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:07:16 2023

molecule_parameters = "config/randomizer_parameters.json"
experimental_data = "config/experimental_data.csv"
output_newmolecules = "molecules/smiles/new_molecules.csv"
output_oldmolecules = "molecules/smiles/old_molecules.csv"

@author: ivanp
"""
import argparse
import json
import urllib
import time
import numpy as np
import pandas as pd

#inner constants - subjected to change
NOM_DIC = {2:"di",3:"tri",4:"tetra",5:"penta",6:"hexa",7:"septa"}
MAX_SUBS = max(NOM_DIC.keys())

def convert_to_smiles(ids):
    "get SMILES form from molecule name"
    url = 'http://cactus.nci.nih.gov/chemical/structure/' + urllib.parse.quote(ids) + '/smiles'
    try:
        ans = urllib.request.urlopen(url,timeout=20).read().decode('utf8')
    except urllib.request.HTTPError:
        print(f"{ids} has no smiles representation")
        ans = ""
    except urllib.request.URLError:
        print("Connection timed-out, sleeping and retrying")
        time.sleep(60)
        ans = convert_to_smiles(ids)
    return ans

def generate_molecule(positions,min_sub,max_sub,subs,base=None):
    """
    generates a molecule

    positions : iterator of ints
        All positions on molecule
    min_sub : int
        Minimum number of substituents
    max_sub : int
        Maximum number of substituents
    subs : iterator of strings
        All substituents
    base:
        Base of molecule

    """
    #print(positions,min_sub,max_sub,subs,base)
    n_subs = np.random.choice(range(min_sub,max_sub+1))
    pos = np.random.choice(positions,n_subs,replace=False)

    substituents = np.random.choice(subs,n_subs).tolist()

    dataframe = pd.DataFrame([substituents,pos],index=["s","p"]).T
    dataframe = dataframe.sort_values("p",ascending=True)
    names = list()
    for sub,pos in dataframe.groupby("s"):
        pos = sorted(pos["p"].tolist())
        if len(pos)==1:
            names.append(f"{pos[0]}-{sub}")
            continue
        position = ",".join([str(x) for x in pos])
        prefix = NOM_DIC[len(pos)]
        names.append(f"{position}-{prefix}{sub}")
    final = ",".join(names)
    if base:
        final = final + f" {base}"
    return final

def generate_molecules(base,positions,min_sub,max_sub,subs,n_molecules,old_molecules):
    """
    generates tons of molecules

    base: str
        Base of molecule
    positions : iterator of ints
        All positions on molecule
    min_sub : int
        Minimum number of substituents
    max_sub : int
        Maximum number of substituents
    subs : iterator of strings
        All substituents
    n_molecules : int
        Maximum number of generated molecules
    old_molecules : iterator of str
        Molecules that will not be created

    """
    _new_molecules = set()

    safety_limit = 100
    i=0
    while len(_new_molecules)<n_molecules:

        if i>safety_limit:
            print("Safety limit reached - likely no new molecules")
            break

        molecule = generate_molecule(positions,min_sub,max_sub,subs,base)

        if molecule in _new_molecules or molecule in old_molecules:
            i=i+1
            continue

        _new_molecules.add(molecule)
        i=0

    return _new_molecules

def smilerize_molecules(experimental_data,molecule_parameters,output_newmolecules,output_oldmolecules):
    """
    experimental_data : path to CSV file
        First column must be names of the compounds
    molecule_parameters : path to JSON file
        JSON file must contian information about new molecules.
        This should include:
            base - base molecule ("benzene","coumarin",etc)
            positions - list of integers that represent substitution positions
            min_sub - minimum number of substituents
            max_sub - maximum number of substituents
            subs - all valid substituents
    output_newmolecules : path to CSV file
    output_oldmolecules : path to CSV file
    """
    #load experimental molecules
    experimental_molecules = set(pd.read_csv(experimental_data,header=None)[0])

    #load parameters
    with open(molecule_parameters, "r") as _f:
        parameters = json.load(_f)

    #mainloop for extension
    new_molecules = set()
    for molinfo in parameters.values():
        molinfo["max_sub"] = min(MAX_SUBS,molinfo["max_sub"])
        molecules = generate_molecules(old_molecules=experimental_molecules,**molinfo)
        new_molecules = new_molecules.union(molecules)

    new_molecules_names = sorted(list(new_molecules))
    old_molecules_names = sorted(list(experimental_molecules))

    #convert to SMILES
    new_molecules = dict()
    old_molecules = dict()

    for name in new_molecules_names:
        smiles = convert_to_smiles(name)
        if smiles:
            new_molecules[name] = smiles
    for name in old_molecules_names:
        old_molecules[name] = convert_to_smiles(name)

    #save
    pd.DataFrame([new_molecules]).T.to_csv(output_newmolecules,header=False)
    pd.DataFrame([old_molecules]).T.to_csv(output_oldmolecules,header=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-x', '--experimental_data', default="config/experimental_data.csv")
    parser.add_argument('-p', '--molecule_parameters', default="config/randomizer_parameters.json")
    parser.add_argument('-o1', '--output_newmolecules', default="molecules/smiles/new_molecules.csv")
    parser.add_argument('-o2', '--output_oldmolecules', default="molecules/smiles/old_molecules.csv")

    smilerize_molecules(**vars(parser.parse_args()))