from math import cos, sin, sqrt, acos, atan2, fabs, pi    
import numpy as np
import math

########################################################
def find_BA(dd1, dd2, dd3, dd4):

    angleID = -1
    for aa in range(len(angles)):
        if ((dd2, dd3, dd4) == angles[aa]) or ((dd4, dd3, dd2) == angles[aa]):
            angleID = aa
            break
    if (angleID == -1):
        print("angle not found", dd2, dd3, dd4)
        exit()
    #find bond
    bondID = -1
    for bb in range(len(bonds)):
        if ((dd3, dd4) == bonds[bb]) or ((dd4, dd3) == bonds[bb]):
            bondID = bb
            break
    if (bondID == -1):
        print("bond not found")
        print(dd1, dd2, dd3, dd4)
        exit()

    return bondID, angleID

########################################################
def place_atom(atom_a, atom_b, atom_c, angle, torsion, bond):

    #print atom_a, atom_b, atom_c, angle, torsion, bond
    R = bond
    ab = np.subtract(atom_b, atom_a)
    bc = np.subtract(atom_c, atom_b)
    bcn = bc / np.linalg.norm(bc)

    case = 1
    okinsert = False
    while (okinsert == False):
        #case 1
        if (case == 1):
            d = np.array([
                -R * cos(angle), R * cos(torsion) * sin(angle),
                R * sin(torsion) * sin(angle)
            ])
            n = np.cross(bcn, ab)
            n = n / np.linalg.norm(n)
            nbc = np.cross(bcn, n)
        #case 2
        elif (case == 2):
            d = np.array([
                -R * cos(angle), R * cos(torsion) * sin(angle),
                R * sin(torsion) * sin(angle)
            ])
            n = np.cross(ab, bcn)
            n = n / np.linalg.norm(n)
            nbc = np.cross(bcn, n)
        #case 3
        elif (case == 3):
            d = np.array([
                -R * cos(angle), R * cos(torsion) * sin(angle),
                -R * sin(torsion) * sin(angle)
            ])
            n = np.cross(ab, bcn)
            n = n / np.linalg.norm(n)
            nbc = np.cross(bcn, n)
        #case 4
        elif (case == 4):
            d = np.array([
                -R * cos(angle), R * cos(torsion) * sin(angle),
                R * sin(torsion) * sin(angle)
            ])
            n = np.cross(ab, bcn)
            n = n / np.linalg.norm(n)
            nbc = np.cross(n, bcn)
            
        # 得到以b为原点的垂直三维坐标系
        m = np.array([[bcn[0], nbc[0], n[0]], [bcn[1], nbc[1], n[1]],
                        [bcn[2], nbc[2], n[2]]])
        d = m.dot(d)
        atom_d = d + atom_c

        #test dihedral
        r21 = np.subtract(atom_b, atom_a)
        r23 = np.subtract(atom_b, atom_c)
        r43 = np.subtract(atom_d, atom_c)
        n1 = np.cross(r21, r23)
        n2 = np.cross(r23, r43)
        n1 = n1 / np.linalg.norm(n1)
        n2 = n2 / np.linalg.norm(n2)
        r23 = r23 / np.linalg.norm(r23)
        m = np.cross(n1, r23)
        x = np.dot(n1, n2)
        y = np.dot(m, n2)
        phi = atan2(y, x)

        #test angle
        d12 = np.subtract(atom_b, atom_c)
        d32 = np.subtract(atom_d, atom_c)
        d12 = d12 / np.linalg.norm(d12)
        d32 = d32 / np.linalg.norm(d32)
        cos_theta = np.dot(d12, d32)
        m = np.linalg.norm(np.cross(d12, d32))
        theta = atan2(m, cos_theta)

        if (fabs(theta - angle) < 0.001 and fabs(phi - torsion) < 0.001):
            okinsert = True
        else:
            if (case < 4): case += 1
            else:
                print("no case found", theta, angle, phi, torsion, atom_d)
                break
    return atom_d

########################################################
def test_angle(atoms, angleID):
    ii, jj, kk = angles[angleID]
    d12 = np.subtract(atoms[ii], atoms[jj])
    d32 = np.subtract(atoms[kk], atoms[jj])
    d12 = d12 / np.linalg.norm(d12)
    d32 = d32 / np.linalg.norm(d32)
    cos_theta = np.dot(d12, d32)
    m = np.linalg.norm(np.cross(d12, d32))
    theta = atan2(m, cos_theta)

    return theta

########################################################
def test_dihedral(atoms, dihedralID):

    ii, jj, kk, ll = dih[dihedralID]
    r21 = np.subtract(atoms[jj], atoms[ii])
    r23 = np.subtract(atoms[jj], atoms[kk])
    r43 = np.subtract(atoms[ll], atoms[kk])

    n1 = np.cross(r21, r23)
    n2 = np.cross(r23, r43)

    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    r23 = r23 / np.linalg.norm(r23)

    m = np.cross(n1, r23)
    x = np.dot(n1, n2)
    y = np.dot(m, n2)

    phi = atan2(y, x)

    return phi


def new_config(CVsB, CVsA, CVsD):

    ang = CVsA[0]

    an = -1.0 * ang
    R1 = np.array([[cos(an), -sin(an), 0.0], [sin(an),
                                                cos(an), 0.0],
                    [0.0, 0.0, 1.0]])
    R2 = np.array([[1.0, 0.0, 0.0],
                    [0.0, cos(-math.pi / 4), -sin(-math.pi / 4)],
                    [0.0, sin(-math.pi / 4),
                    cos(-math.pi / 4)]])
    R3 = np.array([[cos(-math.pi / 4), 0.0,
                    sin(-math.pi / 4)], [0.0, 1.0, 0.0],
                    [-sin(-math.pi / 4), 0.0,
                    cos(-math.pi / 4)]])

    atoms = np.zeros((154, 3), float)

    ### first 3 atoms ###
    vec01 = [1.0 / sqrt(2), 1.0 / sqrt(2), 0.0]
    vec31 = np.dot(R1, vec01)
    vec01 = np.dot(R2, vec01)
    vec31 = np.dot(R2, vec31)
    vec01 = np.dot(R3, vec01)
    vec31 = np.dot(R3, vec31)

    atoms[0] = [CVsB[0] * vec01[0], CVsB[0] * vec01[1], CVsB[0] * vec01[2]]
    atoms[1] = [0.0, 0.0, 0.0]
    atoms[2] = [CVsB[1] * vec31[0], CVsB[1] * vec31[1], CVsB[1] * vec31[2]]

    ### iteratively all other atoms ###
    find_atoms = [0, 1, 2]
    tmp_dihedrals = [(idx, dihedral) for idx, dihedral in enumerate(dih)]
    while tmp_dihedrals:
        # 必须保证前三个坐标都确定了，而且最后一个原子不在已经找到的原子里，否则放在队列里
        (dd, dihedral) = tmp_dihedrals.pop(0)
        dd1, dd2, dd3, dd4 = dihedral
        if all(a in find_atoms for a in dihedral[:3]):
            if dd4 in find_atoms:
                print(dihedral)   
            bondID, angleID = find_BA(dd1, dd2, dd3, dd4)
            coord = place_atom(atoms[dd1], atoms[dd2], atoms[dd3],
                                CVsA[angleID], CVsD[dd], CVsB[bondID])
            atoms[dd4] = coord
            find_atoms.append(dd4)
        else:
            tmp_dihedrals.append((dd, dihedral))
        
    testBAD = False
    if (testBAD):
        #bonds
        for mm in range(len(bonds)):
            ii, jj = bonds[mm]
            dist = pow(atoms[ii][0] - atoms[jj][0], 2) + pow(
                atoms[ii][1] - atoms[jj][1], 2) + pow(
                    atoms[ii][2] - atoms[jj][2], 2)
            if (fabs(sqrt(dist) - CVsB[mm]) > 0.0001):
                print("bond", bonds[mm], CVsB[mm], sqrt(dist), atoms[ii],
                        atoms[jj], "Reading snapshot ", Nfile)
        #angles
        for mm in range(len(angles)):
            acos_theta = test_angle(atoms, mm)
            #print "angle",angles[mm],CVsA[mm]*180/pi,acos_theta*180/pi
            if (fabs(acos_theta - CVsA[mm]) > 0.0001):
                print("angle", angles[mm], CVsA[mm], acos_theta,
                        "Reading snapshot ", Nfile)
        #dihedrals
        for mm in range(len(dih)):
            acos_theta = test_dihedral(atoms, mm)
            #print "dihedral",dih[mm],CVsD[mm]*180/pi,acos_theta*180/pi
            if (fabs(acos_theta - CVsD[mm]) > 0.0001):
                print("dihedral", dih[mm], CVsD[mm], acos_theta,
                        "Reading snapshot ", Nfile)

    return atoms


if __name__ == "__main__":
    import pickle as pk
    import MDAnalysis as mda
    from tqdm import tqdm
    import pdb
    pdb.set_trace()
    
    DataFilePath = '/workspace/lizt/AI_molucular_dynamics_simulation/Modified_LED-Molecular/Results/TRP/Evaluation_Data/GPU-ARNN-scaler_MinMaxZeroOne-LR_0.001-L2_0.0-MDN_trp-KERN_5-HIDDEN_50-SigmaMax_0.8-DIM_649-AUTO_6x500-ACT_tanh-RES_1-DKP_1-LD_2-C_lstm-R_1x40-SL_200-R-MDN_normal-R-KERN_4-R-HIDDEN_20-R-SMax_0.1/results_iterative_latent_forecasting_train.pickle'
    topolFilePath = '/workspace/lizt/AI_molucular_dynamics_simulation/Modified_LED-Molecular/Data/TRP/topol.top'
    
    
    res = pk.load(open(DataFilePath, 'rb'))
    trajs = res["predictions_all"].reshape(-1, 649)
    traj = mda.Universe(topolFilePath, topology_format='ITP')

    # 从全原子拓扑图中生成粗粒度原子的universe文件，用于轨迹生成
    # construct a universe with 154  heavy atoms in residues
    resindices = [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,
        2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,
        3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  5,
        5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  7,
        7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  9,
        9,  9,  9, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12,
       12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15,
       15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17,
       17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19,
       19]

    u = mda.Universe.empty(154, 20, atom_resindex=resindices, trajectory=True)
    
    atoms = traj.select_atoms("protein").select_atoms("type N CW C CA CT NA O CB N2 O2 CN N3 C* OH")
    ori_target_map = dict(zip(atoms.indices, range(atoms.n_atoms)))
    target_bonds = [(ori_target_map[bond[0]], ori_target_map[bond[1]]) for bond in atoms.bonds.indices if all(a in atoms.indices for a in bond)]
    target_angles = [(ori_target_map[angle[0]], ori_target_map[angle[1]], ori_target_map[angle[2]]) for angle in atoms.angles.indices if all(a in atoms.indices for a in angle)]
    target_dihedrals = [(ori_target_map[dihedral[0]], ori_target_map[dihedral[1]], ori_target_map[dihedral[2]], ori_target_map[dihedral[3]]) for dihedral in atoms.dihedrals.indices if all(a in atoms.indices for a in dihedral)]
 
    
    u.add_TopologyAttr('bonds', target_bonds)
    u.add_TopologyAttr('angles', target_angles)
    u.add_TopologyAttr('dihedrals', target_dihedrals)
    
    
    bond_dim = 160
    angle_dim = 219
    dehedral_dim = 270	
    bond_indices = bond_dim
    angle_indices = bond_dim + angle_dim 
    
    # construct 1000000 frame, 154 atoms, xyz coordinates
    
    dih = ((0,1,2,3), (1,2,3,4), (1,2,3,5), (3,2,1,6), (0,1,6,7), (0,1,6,8), (1,6,8,9), (6,8,9,10), (8,9,10,11), (9,10,11,12), (9,10,11,13), (6,8,9,14), \
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
    angles = ((0,1,2), (1,2,3), (2,3,4), (2,3,5), (2,1,6), (1,6,7), (1,6,8), (6,8,9), (8,9,10), (9,10,11), (10,11,12), (10,11,13), (8,9,14), (9,14,15), \
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
    bonds = ((0,1), (1,2), (2,3), (3,4), (3,5), (1,6), (6,7), (6,8), (8,9), (9,10), (10,11), (11,12), (11,13), (9,14), (14,15), (14,16), (16,17), (17,18), \
    (18,19), (19,20), (20,21), (21,22), (22,23), (22,24), (19,25), (17,26), (26,27), (26,28), (28,29), (29,30), (30,31), (30,32), (32,33), (29,34), (34,35), \
    (34,36), (36,37), (37,38), (38,39), (39,40), (40,41), (40,42), (37,43), (43,44), (43,45), (45,46), (46,47), (47,48), (48,49), (49,50), (50,51), (51,52), \
    (52,53), (53,54), (54,55), (48,56), (46,57), (57,58), (57,59), (59,60), (60,61), (61,62), (62,63), (62,64), (60,65), (65,66), (65,67), (67,68), (68,69), \
    (69,70), (70,71), (71,72), (72,73), (68,74), (74,75), (74,76), (76,77), (77,78), (78,79), (79,80), (79,81), (77,82), (82,83), (82,84), (84,85), (85,86), \
    (86,87), (86,88), (88,89), (89,90), (90,91), (90,92), (92,93), (93,94), (94,95), (92,96), (96,97), (97,98), (97,99), (99,100), (100,101), (101,102), \
    (100,103), (103,104), (103,105), (105,106), (106,107), (107,108), (106,109), (109,110), (109,111), (111,112), (112,113), (113,114), (113,115), (115,116), \
    (116,117), (117,118), (118,119), (119,120), (120,121), (121,122), (121,123), (116,124), (124,125), (124,126), (126,127), (127,128), (128,129), (126,130), \
    (130,131), (131,132), (131,133), (133,134), (134,135), (135,136), (133,137), (137,138), (138,139), (138,140), (140,141), (141,142), (142,143), (140,144), \
    (144,145), (145,146), (145,147), (147,148), (148,149), (149,150), (148,151), (151,152), (151,153))

    

    
    nconflict_b_indices = []
    for b in bonds:
        nconflict_b_indices.append(target_bonds.index(b))
    
    nconflict_a_indices = []
    for a in angles:
        nconflict_a_indices.append(bond_indices+target_angles.index(a))        
    
    nconflict_d_indices = []
    for d in dih:
        nconflict_d_indices.append(angle_indices+target_dihedrals.index(d))
    
    
    def get_coordinates(start_end):
        coords = []
        start, end = start_end
        for Nfile, frame in tqdm(enumerate(trajs[start:end])):
            coords.append(new_config(frame[nconflict_b_indices], frame[nconflict_a_indices], frame[nconflict_d_indices]))
        return coords
    

    
    from multiprocessing import Pool, cpu_count
    import math
    from functools import partial
    
    
    # num_cores = cpu_count()
    num_cores  = 1
    num_steps = len(trajs) 
    num_steps = 10
    if num_cores > 1:
        num_state_per_state = math.ceil(num_steps / num_cores) 
        p = Pool(num_cores)
        start_points = [i*num_state_per_state for i in range(num_cores)]
        end_points = [min(num_steps, (i+1)*num_state_per_state) for i in range(num_cores)]
        results = p.map(get_coordinates, zip(start_points, end_points))
        p.close()
        p.join()
        coordinates = []
        for res in results:
            coordinates.extend(res)
    else:
        coordinates = get_coordinates((0, num_steps))
        
    coordinates = np.array(coordinates)
    
    # 计算分子重心并计算各个原子的距离重心的距离
    def remove_com(conf):
        # calculate center of mass165
        comp = [0.0, 0.0, 0.0]
        masstotal = sum(mass)
        for i in range(len(conf)):
            for dim in range(3):
                comp[dim] += mass[i] * conf[i][dim]
        for dim in range(3):
            comp[dim] /= masstotal

        # substract center of mass
        conf_com = np.zeros((len(conf), 3), float)
        for i in range(len(conf)):
            for dim in range(3):
                conf_com[i, dim] = conf[i][dim] - comp[dim]

        return conf_com


    # 根据参考分子的角度旋转现在的分子角度
    def rotationmatrix(coordref, coord):

        assert (coordref.shape[1] == 3)
        assert (coordref.shape == coord.shape)
        correlation_matrix = np.dot(np.transpose(coordref), coord)
        vv, ss, ww = np.linalg.svd(correlation_matrix)
        is_reflection = (np.linalg.det(vv) * np.linalg.det(ww)) < 0.0
        #if is_reflection:
        #print "is_reflection"
        #vv[-1,:] = -vv[-1,:]
        #ss[-1] = -ss[-1]
        #vv[:, -1] = -vv[:, -1]
        rotation = np.dot(vv, ww)

        confnew = []
        for i in range(len(coord)):
            xx = rotation[0][0] * coord[i][0] + rotation[0][1] * coord[i][
                1] + rotation[0][2] * coord[i][2]
            yy = rotation[1][0] * coord[i][0] + rotation[1][1] * coord[i][
                1] + rotation[1][2] * coord[i][2]
            zz = rotation[2][0] * coord[i][0] + rotation[2][1] * coord[i][
                1] + rotation[2][2] * coord[i][2]
            confnew.append((xx, yy, zz))

        return confnew
    
    # 记录154个原子的重量
    mass = atoms.masses
    conf_ref = remove_com(coordinates[0])
    rotation_coordinates = [conf_ref]
    for i in range(1, coordinates.shape[0]):
        conf_com = remove_com(coordinates[i])
        rotation_coordinates.append(rotationmatrix(conf_ref, conf_com))
     
    rotation_coordinates = np.array(rotation_coordinates)
    all_coordinates = np.zeros((num_steps, traj.atoms.n_atoms, 3))
    
    all_coordinates[:, atoms.indices, :] = rotation_coordinates
    traj.load_new(all_coordinates, in_memory=True)
    traj.dimensions = [500, 500, 500, 90, 90, 90]
    traj.select_atoms('protein').select_atoms("type N CW C CA CT NA O CB N2 O2 CN N3 C* OH").write('trp.xtc', frames='all')