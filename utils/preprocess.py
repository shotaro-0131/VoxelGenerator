import math
# from rdkit.Chem import AllChem
# from rdkit import Chem
import json
import gzip
import os.path
import sys
import numpy as np
# from Bio import PDB
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import warnings
# from rdkit import RDLogger
import cmd
import pymol
GRID_LENGTH = 20


def random_rotation_3d(batch, max_angle):
    size = batch.shape
    batch = np.squeeze(batch)
    batch_rot = np.zeros(batch.shape)
    for i in range(batch.shape[0]):
        if bool(random.getrandbits(1)):
            image1 = np.squeeze(batch[i])
            # rotate along z-axis
            angle = random.uniform(-max_angle, max_angle)
            image2 = scipy.ndimage.interpolation.rotate(
                image1, angle, mode='nearest', axes=(0, 1), reshape=False)

            # rotate along y-axis
            angle = random.uniform(-max_angle, max_angle)
            image3 = scipy.ndimage.interpolation.rotate(
                image2, angle, mode='nearest', axes=(0, 2), reshape=False)

            # rotate along x-axis
            angle = random.uniform(-max_angle, max_angle)
            batch_rot[i] = scipy.ndimage.interpolation.rotate(
                image3, angle, mode='nearest', axes=(1, 2), reshape=False)

        else:
            batch_rot[i] = batch[i]
    return batch_rot.reshape(size)


atoms_info = {
    'C': 0,
    'O': 1,
    'N': 2,
    'S': 3,
    'P': 4,
    'CL': 5
}


def filter_ligands(ligand):
    if all(a.GetSymbol().upper() in atoms_info for a in ligand.GetAtoms()):
        yield r
    # return


def filter_mol_atoms(mol):
    for a in mol:
        if a.GetSymbol() in atoms_info:
            yield a


def filter_atoms(atoms):
    for a in atoms:
        if a[3] in atoms_info:
            yield a
    # raise StopIteration


def calc_center(residue):
    acc = np.zeros(3)
    i = 0
    for r in residue.get_residues():
        for a in r.get_atom():
            acc += a.get_coord()
            i += 1
    return acc / i


def calc_center_ligand(mol):
    acc = np.zeros(3)
    i = 0
    for m in mol:
        x, y, z, _ = m
        p = [x, y, z]
        acc += p
        i += 1
    return acc / i


def get_atom_index(atom):
    return atoms_info[atom]


def centering(structure, center, square_length=20):
    points = []

    for a in filter_atoms(structure):
        x, y, z, atom = a
        p = [x, y, z] - center
        if all((np.abs(p[i]) < square_length/2 for i in range(3))):
            points.append(np.concatenate([p, [get_atom_index(atom)]]))
    return points

def proc_structure(pocket, ligand, grid_length):
    if ligand != None:
        ligand_center = calc_center_ligand(ligand)
        yield centering(pocket, ligand_center, grid_length), centering(ligand, ligand_center, grid_length)


def read_pdb(filename):
    warnings.filterwarnings('ignore')
    base, ext = os.path.splitext(filename)
    if ext == '.pdb':
        fp = open(filename)
    elif ext == '.gz' and os.path.splitext(base)[1] == '.pdb':
        fp = gzip.open(filename, 'rt')
    else:
        raise Exception()
    p = PDB.PDBParser()
    s = p.get_structure('', fp)
    fp.close()
    return s


def get_mol(filename):
    # RDLogger.DisableLog('rdApp.*')
    warnings.filterwarnings('ignore')
    cmd = pymol.cmd
    base, ext = os.path.splitext(filename)
    mol = []
    
    if ext == '.mol2' or ext == '.sdf' or ext == ".pdb":
        cmd.delete("all")
        cmd.load(filename)
        cmd.iterate_state(1, 'all', 'mol.append([x, y, z, elem])', space=locals(), atomic=0)
    else:
        raise Exception()
    return mol


def get_points(protein_file_path, ligand_file_path, n_cell=20, cell_size=1):
    grid_length = n_cell*cell_size
    # parser = PDB.PDBParser()
    pdb_path = protein_file_path
    mol2_path = ligand_file_path
    protein_points = []
    ligand_points = []
    if os.path.exists(pdb_path) and os.path.exists(mol2_path):
        p = get_mol(pdb_path)
        l = get_mol(mol2_path)
        # p_mol = get_mol(pdb_path)
        for d in proc_structure(p, l, grid_length):
            protein_points, ligand_points = d
        return protein_points, ligand_points

def xyz(array, p, cell_size=1):
    _, X, Y, Z, _ = array.shape
    i = int((p[0] + X*cell_size/2)/cell_size)
    j = int((p[1] + Y*cell_size/2)/cell_size)
    k = int((p[2] + Z*cell_size/2)/cell_size)
    atom_index = int(p[3])
    array[atom_index, i, j, k] = [(p[0] + X*cell_size/2)-i*cell_size, (p[1] + Y*cell_size/2)-j*cell_size, (p[2] + Z*cell_size/2)-k*cell_size]

def fill_cell(array, p, cell_size=1):
    _, X, Y, Z = array.shape
    i = int((p[0] + X*cell_size/2)/cell_size)
    j = int((p[1] + Y*cell_size/2)/cell_size)
    k = int((p[2] + Z*cell_size/2)/cell_size)
    atom_index = int(p[3])
    for c in range(8):
        ii, jj, kk = 0, 0, 0
        if c >> 0 & 1 == 1 and i != X-1:
            ii = 1
        if c >> 1 & 1 == 1 and j != Y-1:
            jj = 1
        if c >> 2 & 1 == 1 and k != Z-1:
            kk = 1
        array[atom_index, i+ii, j+jj, k+kk] = 1


def to_voxel(points, n_cell=20, cell_size=1):
    voxel = np.zeros(n_cell**3 * len(atoms_info), dtype=np.int8)\
        .reshape([len(atoms_info), n_cell, n_cell, n_cell])
    for point in points:
        fill_cell(voxel, point, cell_size)
    return voxel

def to_delta_xyz(points, n_cell=20, cell_size=1):
    voxel = np.zeros(n_cell**3 * len(atoms_info) * 3, dtype=np.int8)\
        .reshape([len(atoms_info), n_cell, n_cell, n_cell, 3])
    for point in points:
        xyz(voxel, point, cell_size)
    return voxel

def process_grid(array, grid_size=20, hreshold=0.2):
    new_array = np.zeros((6, grid_size, grid_size, grid_size), dtype=bool)
    for i in range(3):
        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(grid_size):
                    new_array[i, x, y, z] = True if array[i,
                                                          x, y, z] > hreshold else False
    return new_array


# %matplotlib inline
def savegrid(d, filename, grid_size=20, hreshold=0.2):
    # prepare some coordinates
    d = process_grid(d, grid_size, hreshold=hreshold)
    x, y, z = np.indices((grid_size, grid_size, grid_size))

    # combine the objects into a single boolean array
    voxels = d[0] | d[1] | d[2] | d[3] | d[4] | d[5]

    # set the colors of each object
    colors = np.empty(voxels.shape, dtype=object)
    colors[d[0]] = 'gray'
    colors[d[1]] = 'red'
    colors[d[2]] = 'blue'
    colors[d[3]] = 'green'
    colors[d[4]] = 'pink'
    colors[d[5]] = 'yellow'

    # and plot everything
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
    plt.savefig(filename, dpi=140)

def score_grid(true_points, target_grid, is_rotation=True):
    for i in range(target_grid.shape[0]):
        if bool(random.getrandbits(1)):
            image1 = np.squeeze(batch[i])
            # rotate along z-axis
            angle = random.uniform(-max_angle, max_angle)
            image2 = scipy.ndimage.interpolation.rotate(
                image1, angle, mode='nearest', axes=(0, 1), reshape=False)

            # rotate along y-axis
            angle = random.uniform(-max_angle, max_angle)
            image3 = scipy.ndimage.interpolation.rotate(
                image2, angle, mode='nearest', axes=(0, 2), reshape=False)

            # rotate along x-axis
            angle = random.uniform(-max_angle, max_angle)
            batch_rot[i] = scipy.ndimage.interpolation.rotate(
                image3, angle, mode='nearest', axes=(1, 2), reshape=False)

        else:
            batch_rot[i] = batch[i]

import numpy as np
import math as m
  
def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])
  
def Ry(theta):
  return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(theta), 0, m.cos(theta)]])
  
def Rz(theta):
  return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

def rotate(points, phi, theta, psi):
    ps, atoms = shaping(points)
    R = Rz(psi) * Ry(theta) * Rz(phi)
    outputs = []
    for i in range(len(ps)):
        t = np.array(R * ps[i]).reshape(1,-1).tolist()[0]
        t.append(atoms[i])
        outputs.append(t)
    return outputs

