#External Modules
import os, ase, json, ase.io
import numpy as np
import pdb
from torch.utils.data import Dataset
#Internal Modules

class CNNInputDataset(Dataset):
    """
    Subclass of Torch Dataset object
    storage_directories :: list of storage directories where chargemol analysis,
                            result.json, and final.traj are stored
    """
    def __init__(self,storage_directories):
        self.storage_directories = storage_directories

    def __getitem__(self, index):
        return self.storage_to_CNN_input(self.storage_directories[index])

    def __len__(self):
        return len(self.storage_directories)

    def storage_to_CNN_input(self, stordir):
        """
        Take a storage directory, with chargemol_analysis and job_output subfolders,
        and produce a connectivity matrix
        """
        results             = json.load(open(stordir+'/job_output/result.json'))
        energy              = results['raw_energy']
        #Extract the connectivity
        connectivity        = self.stordir_to_connectivity(stordir)
        #Get the node feature matrix
        atoms_obj           = self.stordir_to_atoms(stordir)
        node_feature_matrix = self.atoms_to_node_features(atoms_obj)
        return (connectivity, node_feature_matrix, energy)

    def atoms_to_node_features(self, atoms):
        """
        Converts atoms object into a numpy array
        node_feature_matrix is shape (num_atoms,number_of_features)

        TO CHANGE: number_of_features = 2 (period,group)
        """

        node_feature_matrix = np.zeros((len(atoms),2))
        for (i,atom) in enumerate(atoms):
            node_feature_matrix[i] = self.get_atom_features(atom)
        return node_feature_matrix

    def get_atom_features(self, atom):
        """
        Returns numpy array of atom get_atom_features
        1st iteration: Feature for 1 atom is [period, group]
        """
        period = [0,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3
                  ,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4
                  ,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5
                  ,6,6,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6
                  ,7,7,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]

        group =   [0,1,18,1,2,13,14,15,16,17,18,1,2,13,14,15,16,17,18
                  ,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18
                  ,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18
                  ,1,2,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18
                  ,1,2,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
        atomic_num = atom.number
        return np.array([period[atomic_num],group[atomic_num]])

    def stordir_to_atoms(self, stordir): return ase.io.read(stordir+'/job_output/final.traj')

    def stordir_to_connectivity(self, stordir):
        """
        Uses bonds.json to produce a list, with each element being a list of
        pairs (toAtomIndex, bond order)
        """

        output,suffix = [],'/chargemol_analysis/final/bonds.json'
        with open(stordir+suffix,'r') as f: bond_dicts = json.load(f)
        n = len(self.stordir_to_atoms(stordir))
        for i in range(n):
            newatom = []
            for bond in bond_dicts:
                if bond['fromNode'] == i:
                    if bond['bondorder'] > 0.01:
                        newatom.append((bond['bondorder'],bond['distance'],bond['toNode']))

            sorted_newatom = list(reversed(sorted(newatom))) # bonds in decreasing strength
            maxind = min(12,len(newatom))              # take UP TO 12 bonds
            out_list = [(n,b,d) for b,d,n in sorted_newatom[:maxind]]
            output.append(np.array(out_list))
        return output
