from ase.io import read as aseread
import numpy as np
import math
import h5py
import os


def get_distance(atom1, atom2):
    return np.linalg.norm(atom1-atom2) 

def get_angle(atom1, atom2, atom3):
    a, b = atom2 - atom1, atom3 - atom2
    return math.acos(np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b))*180/math.pi

def get_dihedral(atom1, atom2, atom3, atom4):
    a, b, c = atom2 - atom1, atom3 - atom2, atom4 - atom3
    n1, n2 = np.cross(a, b), np.cross(b, c)

    cos = np.dot(n1, n2)/np.linalg.norm(n1)/np.linalg.norm(n2)
    cos = max(-1, min(1, cos))
    return math.acos(cos)*180/math.pi
    

def writeToHDF5Files(sequences, data_dir_str):
    data_dir = "./Data/{:}".format(data_dir_str)
    os.makedirs(data_dir, exist_ok=True)
    hf = h5py.File(data_dir + '/data.h5', 'w')
    # Only a single sequence_example per dataset group
    for seq_num_ in range(np.shape(sequences)[0]):
        print('batch_{:010d}'.format(seq_num_))
        data_group = sequences[seq_num_]
        data_group = np.array(data_group)
        # print(np.shape(data_group))
        gg = hf.create_group('batch_{:010d}'.format(seq_num_))
        gg.create_dataset('data', data=data_group)
    hf.close()

if __name__ == "__main__":
    import MDAnalysis as mda
    # import pdb 
    # pdb.set_trace()
    
    # load top file
    from tqdm import tqdm 
    traj = mda.Universe('topol.top', 'AA-fixed.xtc', topology_format='ITP')
    
    atoms = traj.select_atoms('protein')
    
    # filter hydrogen
    valid_atom_types = ['N3', 'CT', 'C', 'O', 'N', 'O2']
    valid_atoms_indices = []
    for atom_type in valid_atom_types:
        valid_atoms_indices.extend(atoms.select_atoms('type %s' % atom_type).indices.tolist())

    valid_bonds = []
    bonds_indices = traj.bonds.to_indices()
    for bond in bonds_indices:
        if all(atom in valid_atoms_indices for atom in bond):
            valid_bonds.append(bond)
    
    
    valid_angles = []
    angles_indices = traj.angles.to_indices()
    for angle in angles_indices:
        if all(atom in valid_atoms_indices for atom in angle):
            valid_angles.append(angle)
    
    valid_dihedrals = []
    dihedrals_indices = traj.dihedrals.to_indices()
    for dihedral in dihedrals_indices:
        if all(atom in valid_atoms_indices for atom in dihedral):
            valid_dihedrals.append(dihedral) 
    states = []
 
    for ts in tqdm(traj.trajectory):
        state = []
        positions = atoms.positions
        # bonds
        for (atom1, atom2) in valid_bonds:
            state.append(get_distance(positions[atom1], positions[atom2]))
        # angle
        for (atom1, atom2, atom3) in valid_angles:
            state.append(get_angle(positions[atom1], 
                                         positions[atom2], 
                                         positions[atom3]))
        # dihedral
        for (atom1, atom2, atom3, atom4) in valid_dihedrals:
            state.append(get_dihedral(positions[atom1], 
                                         positions[atom2], 
                                         positions[atom3],
                                         positions[atom4]))
        states.append(state)
    # split train val test
    time_per_iter = 0.1 # ps
    time_per_sample = 400 # ps
    iter_per_sample = int(time_per_sample/time_per_iter)
    init_iteration = 8000
    bonds_dim = len(valid_bonds)
    angle_dim = len(valid_angles)
    dihedral_dim = len(valid_dihedrals)
    # reshape = 
    num_steps = len(states)
    num_sample = int((num_steps-init_iteration)/iter_per_sample)
    states = np.array(states[init_iteration:num_sample*iter_per_sample+init_iteration])
    states = states.reshape((-1, int(iter_per_sample), bonds_dim+angle_dim+dihedral_dim))
    num_train = 96
    num_val = 96
    num_test = 248
    
    # save h5 file
    data = states[:num_train]
    writeToHDF5Files(data, "train")

    data = states[num_train:num_train+num_val]
    writeToHDF5Files(data, "val")

    data = states[:num_test]
    writeToHDF5Files(data, "test")


    min_bond = states[:, :, :bonds_dim].min(axis=(0,1))
    max_bond = states[:, :, :bonds_dim].max(axis=(0,1))
    min_angle = states[:, :, bonds_dim:bonds_dim+angle_dim].min(axis=(0,1))
    max_angle = states[:, :, bonds_dim:bonds_dim+angle_dim].max(axis=(0,1))
    np.savetxt('./Data/data_min_bonds.txt', min_bond)
    np.savetxt('./Data/data_max_bonds.txt', max_bond)
    np.savetxt('./Data/data_min_angles.txt', min_angle)
    np.savetxt('./Data/data_max_angles.txt', max_angle)
    
    with open('feature_idx.txt', 'w') as f:
        f.write('{}\t{}\t{}\t'.format(bonds_dim, angle_dim, dihedral_dim))
        