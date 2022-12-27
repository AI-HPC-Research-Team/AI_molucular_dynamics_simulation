from ase.io import read as aseread
import numpy as np
import math
import h5py
import os


def get_distance(atom1, atom2):
    return np.linalg.norm(atom1-atom2) 

def get_angle(atom1, atom2, atom3):
    a, b = atom2 - atom1, atom3 - atom2
    return math.acos(np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b))

def get_dihedral(atom1, atom2, atom3, atom4):
    a, b, c = atom2 - atom1, atom3 - atom2, atom4 - atom3
    n1, n2 = np.cross(a, b), np.cross(b, c)

    cos = np.dot(n1, n2)/np.linalg.norm(n1)/np.linalg.norm(n2)
    cos = max(-1, min(1, cos))
    return -np.sign(np.dot(a, b))*math.acos(cos)

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
    import pdb 
    pdb.set_trace()
    
    # load top file
    from tqdm import tqdm 
    traj = mda.Universe('topol.top', 'Trp-Cage-fixed.xtc', topology_format='ITP')
    
    atoms = traj.select_atoms('protein')
    
    # filter hydrogen
    valid_atom_types = ['N', 'CW', 'C', 'CA', 'CT', 'NA', 'O', 'CB', 'N2', 'O2', 'CN', 'N3', 'C*', 'OH']
    # N, CW, C, CA, CT, NA, O, CB, N2, O2, CN, N3, C*, OH
    # valid_atom_types = ['N', 'C', 'O']
    valid_atoms_indices = []
    for atom_type in valid_atom_types:
        valid_atoms_indices.extend(atoms.select_atoms('type %s' % atom_type).indices.tolist())
    valid_atoms_indices = list(set(valid_atoms_indices))
    
    valid_bonds = []
    bonds_indices = traj.bonds.to_indices()
    for bond in bonds_indices:
        if all(atom in valid_atoms_indices for atom in bond):
            valid_bonds.append(bond)
    
    # 原作者论文中漏掉了 7个键 对应topol中的(51, 53)，（105，113），（112，113），（191， 194），（258，261），（272， 275），（286， 289）
    
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
    
    def get_state(start_end):
        states = []
        start_point, end_point = start_end
        copy_traj = traj.copy()
        atoms = copy_traj.select_atoms('protein')
        for ts in tqdm(copy_traj.trajectory[start_point:end_point]):
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
        return states

    from multiprocessing import Pool, cpu_count
    import math
    from functools import partial
    
    
    num_cores = cpu_count()
    # num_cores  = 2
    num_steps = len(traj.trajectory) 
    if num_cores > 1:

        num_state_per_state = math.ceil(num_steps / num_cores) 
        p = Pool(num_cores)
        start_points = [i*num_state_per_state for i in range(num_cores)]
        end_points = [min(num_steps, (i+1)*num_state_per_state) for i in range(num_cores)]
        results = p.map(get_state, zip(start_points, end_points))
        p.close()
        p.join()
        states = []
        for res in results:
            states.extend(res)
    else:
        states = get_state(traj.trajectory)
    # split train val test
    time_per_iter = 0.1 # ps
    time_per_sample = 400 # ps
    iter_per_sample = int(time_per_sample/time_per_iter)
    init_iteration = 8000
    bonds_dim = len(valid_bonds)
    angle_dim = len(valid_angles)
    dihedral_dim = len(valid_dihedrals)
    
    
    num_sample = int((num_steps-init_iteration)/iter_per_sample)
    states = np.array(states[init_iteration:num_sample*iter_per_sample+init_iteration])
    states = states.reshape((-1, int(iter_per_sample), bonds_dim+angle_dim+dihedral_dim))
    print(states.shape)
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
    min_dihedral = states[:, :, bonds_dim+angle_dim:bonds_dim+angle_dim+dihedral_dim].min(axis=(0,1))
    max_dihedral = states[:, :, bonds_dim+angle_dim:bonds_dim+angle_dim+dihedral_dim].max(axis=(0,1))
    np.savetxt('./Data/data_min_bonds.txt', min_bond)
    np.savetxt('./Data/data_max_bonds.txt', max_bond)
    np.savetxt('./Data/data_min_angles.txt', min_angle)
    np.savetxt('./Data/data_max_angles.txt', max_angle)
    np.savetxt('./Data/data_min_dihedral.txt', min_dihedral)
    np.savetxt('./Data/data_max_dihedral.txt', max_dihedral)
    with open('feature_idx.txt', 'w') as f:
        f.write('{}\t{}\t{}\t'.format(bonds_dim, angle_dim, dihedral_dim))
        
