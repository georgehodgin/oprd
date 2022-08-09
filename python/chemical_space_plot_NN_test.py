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

def main():
    
    # Define current working directory so we can navigate folders easily
    os.chdir('../plots/')
    pwd = os.getcwd()
    
    # Load in the data from the Compute_DRFP.ipynb notebook
    data = pickle.load(open('/home/gah/oprd/data/oprd_drfp_with_reagents.pkl',"rb"))
    
    # Isolate drfp fingerprints
    fingerprints =  _np_to_vectorUintd(data[0])

    # Do the same for the query reactions
    query = pickle.load(open('/home/gah/oprd/data/query_reactions_drfp.pkl', "rb"))
    query_drfp = _np_to_vectorUintd(query[0])

    LSH, query_MinHash = LSH_forest_index(fingerprints, query_drfp)

    # query reactions to find 10 nearest neighbours
    NN = LSH.batch_query(query_MinHash,10)

    # Return the nearest neighbour reaction SMILES, Yields, temps, times
    for i in range(len(NN)):

        NN_SMILES = list(data[-1][NN[i]])
        NN_Yields = list(data[1][NN[i]])
        NN_Temps = list(data[3][NN[i]])
        NN_Times = list(data[2][NN[i]])
    
    
        # Store the results
        with open(f"/home/gah/oprd/NN_results/{query[-1][i]}_10_NN.pkl", "wb+") as f:
            pickle.dump((NN_SMILES, NN_Yields, NN_Temps, NN_Times), f, protocol=4)


    # Visualisation of the tmap plot with Faerun
    # Get the x,y coordinates from the LSH forest index
    x, y, s, t = getCoords(LSH)
    
    # Set up a plot with a white background
    faerun = Faerun(view="front", coords=False, clear_color='#FFFFFF')
    # Add the LSH forest data as a scatter plot
    faerun.add_scatter(
        "chemical_space_plot_oprd_with_reagents",
        {   "x": x, 
            "y": y, 
            "c": data[1], # yields
            "labels": data[-1].tolist()}, # SMILES
        point_scale=3,
        colormap = 'Set1_r',
        has_legend=True,
        legend_title = 'Reaction Yield',
        series_title = 'OPRD One Step Reactions',
        categorical=False
        #shader = 'smoothCircle'
    )
    
    # Create the minimum spanning tree
    faerun.add_tree("reactiontree",
                    {"from": s, "to": t},
                    point_helper="chemical_space_plot_oprd_with_reagents")
    
    # Save JS and HTML files for the plot - to be sent to github
    faerun.plot("chemical_space_plot_oprd_with_reagents",
                template="reaction_smiles")
    
def _np_to_vectorUintd(array):
    """ takes the numpy array of a drfp fingerprint and converts it to
        the tmap datatype tmap.VectorUint for use in LSH forest indexing"""
    
    fingerprints = [tmap.VectorUchar(array[i,:]) for i in range(array.shape[0])]
    return fingerprints

def LSH_forest_index(fingerprints, query_fp):
    
    #set # bits i.e. length of MinHash vector
    perm = 512

    # Initialize the Minhash
    enc = tmap.Minhash(d = perm)
    
    # Initialize the LSH Forest
    lf1 = tmap.LSHForest(d= perm, l = 8) # d = the dimensionality of the minhash vectors
                                         # l = number of index tables

    # Add the Fingerprints to the LSH Forest and index
    lf1.batch_add(enc.batch_from_binary_array(fingerprints))
    lf1.index()

    # Create MinHash fingerprints for the query reactions
    q_MinHash = enc.batch_from_binary_array(query_fp)

    return lf1, q_MinHash

    
def getCoords(LSH_forest):

    # Get the coordinates
    x, y, s, t, _ = tmap.layout_from_lsh_forest(LSH_forest)
    return x, y, s, t

if __name__ == "__main__":
    main()