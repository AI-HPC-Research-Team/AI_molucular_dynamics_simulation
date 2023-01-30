import MDAnalysis as mda
from MDAnalysis.analysis import align, rms


if __name__ == "__main__":
    u = mda.Universe('new_topol.gro', 'trp.xtc')
    ref_u = u.copy()
    ref_u.trajectory[0]
    u.trajectory[-1]
    print(ref_u.atoms.positions[:10])
    print(u.atoms.positions[:10])
    print('1: ', u.trajectory.ts)
    unaligned_rmsd = rms.rmsd(u.atoms.positions, ref_u.atoms.positions, superposition=False)
    print('2: ', u.trajectory.ts)
    aligner = align.AlignTraj(u, ref_u, in_memory=True).run()
    u.trajectory[-1]
    print('3: ', u.trajectory.ts)
    aligned_rmsd = rms.rmsd(u.atoms.positions, ref_u.atoms.positions, superposition=False)
    print(unaligned_rmsd, aligned_rmsd)
    # for i in range(len(u.trajectory)):
    # u.trajectory[i]
    # print('1: ', u.trajectory.ts)
    # unaligned_rmsd = rms.rmsd(u.atoms.positions, ref_u.atoms.positions, superposition=False)
    # u.trajectory[i]
    # print('2: ', u.trajectory.ts)
    # aligner = align.AlignTraj(u, ref_u, in_memory=True).run()
    # print('3: ', u.trajectory.ts)
    # aligned_rmsd = rms.rmsd(u.trajectory[i].positions, ref_u.atoms.positions, superposition=False)
    # print(unaligned_rmsd, aligned_rmsd)
    # print(u.atoms.positions[:10])
    