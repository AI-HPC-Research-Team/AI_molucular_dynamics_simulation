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
    return np.sign(np.cross(n1,n2)[-1])*math.acos(cos)

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
    
    # valid_bonds = []
    # bonds_indices = traj.bonds.to_indices()
    # for bond in bonds_indices:
    #     if all(atom in valid_atoms_indices for atom in bond):
    #         valid_bonds.append(bond)
    
    # # 原作者论文中漏掉了 7个键 对应topol中的(51, 53)，（105，113），（112，113），（191， 194），（258，261），（272， 275），（286， 289）
    
    # valid_angles = []
    # angles_indices = traj.angles.to_indices()
    # for angle in angles_indices:
    #     if all(atom in valid_atoms_indices for atom in angle):
    #         valid_angles.append(angle)
    
    # valid_dihedrals = []
    # dihedrals_indices = traj.dihedrals.to_indices()
    # for dihedral in dihedrals_indices:
    #     if all(atom in valid_atoms_indices for atom in dihedral):
    #         valid_dihedrals.append(dihedral) 
    
    valid_dihedrals_p = ((0,1,2,3), (1,2,3,4), (1,2,3,5), (3,2,1,6), (0,1,6,7), (0,1,6,8), (1,6,8,9), (6,8,9,10), (8,9,10,11), (9,10,11,12), (9,10,11,13), (6,8,9,14), \
    (8,9,14,15), (8,9,14,16), (9,14,16,17), (14,16,17,18), (16,17,18,19), (17,18,19,20), (18,19,20,21), (19,20,21,22), (20,21,22,23), (20,21,22,24), (17,18,19,25),\
     (14,16,17,26), (16,17,26,27), (16,17,26,28), (17,26,28,29), (26,28,29,30), (28,29,30,31), (28,29,30,32), (29,30,32,33), (26,28,29,34), (28,29,34,35), (28,29,34,36),\
      (29,34,36,37), (34,36,37,38), (36,37,38,39), (37,38,39,40), (38,39,40,41), (38,39,40,42), (34,36,37,43), (36,37,43,44), (36,37,43,45), (37,43,45,46),\
       (43,45,46,47), (45,46,47,48), (46,47,48,49), (47,48,49,50), (48,49,50,51), (49,50,51,52), (50,51,52,53), (51,52,53,54), (52,53,54,55), (46,47,48,56), \
       (43,45,46,57), (45,46,57,58), (45,46,57,59), (46,57,59,60), (57,59,60,61), (59,60,61,62), (60,61,62,63), (60,61,62,64), (57,59,60,65), (59,60,65,66), \
       (59,60,65,67), (60,65,67,68), (65,67,68,69), (67,68,69,70), (68,69,70,71), (69,70,71,72), (70,71,72,73), (65,67,68,74), (67,68,74,75), (67,68,74,76), \
       (68,74,76,77), (74,76,77,78), (76,77,78,79), (77,78,79,80), (77,78,79,81), (74,76,77,82), (76,77,82,83), (76,77,82,84), (77,82,84,85), (82,84,85,86), \
       (84,85,86,87), (84,85,86,88), (85,86,88,89), (86,88,89,90), (88,89,90,91), (88,89,90,92), (89,90,92,93), (90,92,93,94), (92,93,94,95), (89,90,92,96), \
       (90,92,96,97), (92,96,97,98), (92,96,97,99), (96,97,99,100), (97,99,100,101), (99,100,101,102), (97,99,100,103), (99,100,103,104), (99,100,103,105), \
       (100,103,105,106), (103,105,106,107), (105,106,107,108), (103,105,106,109), (105,106,109,110), (105,106,109,111), (106,109,111,112), (109,111,112,113),\
        (111,112,113,114), (111,112,113,115), (112,113,115,116), (113,115,116,117), (115,116,117,118), (116,117,118,119), (117,118,119,120), (118,119,120,121),\
         (119,120,121,122), (119,120,121,123), (113,115,116,124), (115,116,124,125), (115,116,124,126), (116,124,126,127), (124,126,127,128), (126,127,128,129),\
          (116,124,126,130), (124,126,130,131), (126,130,131,132), (126,130,131,133), (130,131,133,134), (131,133,134,135), (133,134,135,136), (130,131,133,137), \
          (131,133,137,138), (133,137,138,139), (133,137,138,140), (137,138,140,141), (138,140,141,142), (140,141,142,143), (137,138,140,144), (138,140,144,145),\
           (140,144,145,146), (140,144,145,147), (144,145,147,148), (145,147,148,149), (147,148,149,150), (145,147,148,151), (147,148,151,152), (147,148,151,153))
    valid_angles_p = ((0,1,2), (1,2,3), (2,3,4), (2,3,5), (2,1,6), (1,6,7), (1,6,8), (6,8,9), (8,9,10), (9,10,11), (10,11,12), (10,11,13), (8,9,14), (9,14,15), \
    (9,14,16), (14,16,17), (16,17,18), (17,18,19), (18,19,20), (19,20,21), (20,21,22), (21,22,23), (21,22,24), (18,19,25), (16,17,26), (17,26,27), (17,26,28), \
    (26,28,29), (28,29,30), (29,30,31), (29,30,32), (30,32,33), (28,29,34), (29,34,35), (29,34,36), (34,36,37), (36,37,38), (37,38,39), (38,39,40), (39,40,41),\
    (39,40,42), (36,37,43), (37,43,44), (37,43,45), (43,45,46), (45,46,47), (46,47,48), (47,48,49), (48,49,50), (49,50,51), (50,51,52), (51,52,53), (52,53,54), \
    (53,54,55), (47,48,56), (45,46,57), (46,57,58), (46,57,59), (57,59,60), (59,60,61), (60,61,62), (61,62,63), (61,62,64), (59,60,65), (60,65,66), (60,65,67), \
    (65,67,68), (67,68,69), (68,69,70), (69,70,71), (70,71,72), (71,72,73), (67,68,74), (68,74,75), (68,74,76), (74,76,77), (76,77,78), (77,78,79), (78,79,80),\
     (78,79,81), (76,77,82), (77,82,83), (77,82,84), (82,84,85), (84,85,86), (85,86,87), (85,86,88), (86,88,89), (88,89,90), (89,90,91), (89,90,92), (90,92,93), \
     (92,93,94), (93,94,95), (90,92,96), (92,96,97), (96,97,98), (96,97,99), (97,99,100), (99,100,101), (100,101,102), (99,100,103), (100,103,104), (100,103,105), \
     (103,105,106), (105,106,107), (106,107,108), (105,106,109), (106,109,110), (106,109,111), (109,111,112), (111,112,113), (112,113,114), (112,113,115), (113,115,116), \
     (115,116,117), (116,117,118), (117,118,119), (118,119,120), (119,120,121), (120,121,122), (120,121,123), (115,116,124), (116,124,125), (116,124,126), (124,126,127),\
      (126,127,128), (127,128,129), (124,126,130), (126,130,131), (130,131,132), (130,131,133), (131,133,134), (133,134,135), (134,135,136), (131,133,137), (133,137,138),\
       (137,138,139), (137,138,140), (138,140,141), (140,141,142), (141,142,143), (138,140,144), (140,144,145), (144,145,146), (144,145,147), (145,147,148), (147,148,149), \
       (148,149,150), (147,148,151), (148,151,152), (148,151,153))
    valid_bonds_p = ((0,1), (1,2), (2,3), (3,4), (3,5), (1,6), (6,7), (6,8), (8,9), (9,10), (10,11), (11,12), (11,13), (9,14), (14,15), (14,16), (16,17), (17,18), \
    (18,19), (19,20), (20,21), (21,22), (22,23), (22,24), (19,25), (17,26), (26,27), (26,28), (28,29), (29,30), (30,31), (30,32), (32,33), (29,34), (34,35), \
    (34,36), (36,37), (37,38), (38,39), (39,40), (40,41), (40,42), (37,43), (43,44), (43,45), (45,46), (46,47), (47,48), (48,49), (49,50), (50,51), (51,52), \
    (52,53), (53,54), (54,55), (48,56), (46,57), (57,58), (57,59), (59,60), (60,61), (61,62), (62,63), (62,64), (60,65), (65,66), (65,67), (67,68), (68,69), \
    (69,70), (70,71), (71,72), (72,73), (68,74), (74,75), (74,76), (76,77), (77,78), (78,79), (79,80), (79,81), (77,82), (82,83), (82,84), (84,85), (85,86), \
    (86,87), (86,88), (88,89), (89,90), (90,91), (90,92), (92,93), (93,94), (94,95), (92,96), (96,97), (97,98), (97,99), (99,100), (100,101), (101,102), \
    (100,103), (103,104), (103,105), (105,106), (106,107), (107,108), (106,109), (109,110), (109,111), (111,112), (112,113), (113,114), (113,115), (115,116), \
    (116,117), (117,118), (118,119), (119,120), (120,121), (121,122), (121,123), (116,124), (124,125), (124,126), (126,127), (127,128), (128,129), (126,130), \
    (130,131), (131,132), (131,133), (133,134), (134,135), (135,136), (133,137), (137,138), (138,139), (138,140), (140,141), (141,142), (142,143), (140,144), \
    (144,145), (145,146), (145,147), (147,148), (148,149), (149,150), (148,151), (151,152), (151,153))
    
    valid_atoms_indices.sort()
    atoms_map = dict(zip(range(154), valid_atoms_indices))
    
    valid_bonds = []
    for b in valid_bonds_p:
        bond_t = []
        for a in b:
            bond_t.append(atoms_map[a])
        valid_bonds.append(tuple(bond_t))
        
    valid_angles = []
    for angle in valid_angles_p:
        angle_t = []
        for a in angle:
            angle_t.append(atoms_map[a])
        valid_angles.append(tuple(angle_t))

    valid_dihedrals = []
    for d in valid_dihedrals_p:
        dihedral_t = []
        for a in d:
            dihedral_t.append(atoms_map[a])
        valid_dihedrals.append(tuple(dihedral_t))
    
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
        
