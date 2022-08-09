#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 14:38:37 2022

@author: gah
"""


import tmap
from faerun import Faerun
import pickle
import os
import pandas as pd

def main():
    
    # Define current working directory so we can navigate folders easily
    os.chdir('../plots/')
    pwd = os.getcwd()
    
    data = pd.read_csv('/home/gah/oprd/data/oprd_mapped_cores_ECFP_512.csv')
    
    fingerprints =  _np_to_vectorUintd(data.iloc[:,0:-2].to_numpy()) # remember to change this to data[0] for drfps
    
    x, y, s, t = LSH_forest_index(fingerprints)
    
    faerun = Faerun(view="front", coords=False, clear_color='#FFFFFF')
    faerun.add_scatter(
        "chemical_space_plot_oprd_reaction_cores",
        {   "x": x, 
            "y": y, 
            "c": data.iloc[:,-1], # yields
            "labels": data.iloc[:,-2].tolist()}, # SMILES
        point_scale=3,
        colormap = 'Set1_r',
        has_legend=True,
        legend_title = 'Reaction Yield',
        series_title = 'OPRD One Step Reactions',
        categorical=False
        #shader = 'smoothCircle'
    )
    
    faerun.add_tree("reactiontree",
                    {"from": s, "to": t},
                    point_helper="chemical_space_plot_oprd_reaction_cores")
    
    faerun.plot("chemical_space_plot_oprd_reaction_cores",
                template="reaction_smiles")

    
    
def _np_to_vectorUintd(array):
    """ takes the numpy array of a drfp fingerprint and converts it to
        the tmap datatype tmap.VectorUint for use in LSH forest indexing"""
    
    fingerprints = [tmap.VectorUint(array[i,:]) for i in range(array.shape[0])]
    return fingerprints

def LSH_forest_index(fingerprints):
    
    #set # permutations
    perm = 512
    
    # Initialize the LSH Forest
    lf1 = tmap.LSHForest(perm)

    # Add the Fingerprints to the LSH Forest and index
    lf1.batch_add(fingerprints)
    lf1.index()

    # Get the coordinates
    x, y, s, t, _ = tmap.layout_from_lsh_forest(lf1)
    return x, y, s, t

if __name__ == "__main__":
    main()