# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 14:32:12 2023

List of all mordred descriptors is found at
https://mordred-descriptor.github.io/documentation/master/descriptors.html
@author: ivanp
"""

import argparse
import pandas as pd
from mordred import Calculator,descriptors
from rdkit import Chem
from rdkit.Chem import AllChem

def calculate_descriptors(molecule_df,smile_column="smile"):
    """
    Calculate molecular descriptors implemented in mordred package.
    Pass a pandas dataframe whose one column 'smile_column'
    contains SMILES representation
    """

    #embed molecules
    molecule_df["mol"] = molecule_df[smile_column].apply(Chem.MolFromSmiles)
    molecule_df["mol"] = molecule_df.mol.apply(Chem.AddHs)
    molecule_df.mol.apply(AllChem.EmbedMolecule)
    molecule_df.mol.apply(AllChem.MMFFOptimizeMolecule)

    calc = Calculator(descriptors,ignore_3D=False)
    descriptor_df = calc.pandas(molecule_df["mol"])
    descriptor_df.index = molecule_df.index

    #only retain numerical and boolean variables
    _numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64','bool']
    #quickly convert to boolean
    valid_descriptors = descriptor_df.select_dtypes(include=_numerics) * 1

    #dumped_descriptors = [x for x in descriptor_df.columns if x not in valid_descriptors]
    return valid_descriptors

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    data = pd.read_csv(input_path,header=None,index_col=0)
    calculate_descriptors(data,1).to_csv(output_path)
    