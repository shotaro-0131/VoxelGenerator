import math as m
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
import random
import scipy
import scipy.ndimage
from scipy.ndimage.interpolation import shift
def random_rotation_3d(batch, max_angle):
    size = batch[0].shape
    batch_1, batch_2 = batch
    batch_1 = np.squeeze(batch_1)
    batch_2 = np.squeeze(batch_2)
    batch_rot_1 = np.zeros(batch_1.shape)
    batch_rot_2 = np.zeros(batch_2.shape)
    angles = [random.uniform(-max_angle, max_angle) for i in range(3)]

    for i in range(batch_1.shape[0]):
        # if bool(random.getrandbits(1)):
            image1_1 = np.squeeze(batch_1[i])
            image1_2 = np.squeeze(batch_2[i])
            # rotate along z-axis
            image2_1 = scipy.ndimage.interpolation.rotate(
                image1_1, angles[0], mode='nearest', axes=(0, 1), reshape=False)
            image2_2 = scipy.ndimage.interpolation.rotate(
                image1_2, angles[0], mode='nearest', axes=(0, 1), reshape=False)

            # rotate along y-axis
            image3_1 = scipy.ndimage.interpolation.rotate(
                image2_1, angles[1], mode='nearest', axes=(0, 2), reshape=False)
            image3_2 = scipy.ndimage.interpolation.rotate(
                image2_2, angles[1], mode='nearest', axes=(0, 2), reshape=False)
            # rotate along x-axis
            angle = random.uniform(-max_angle, max_angle)
            batch_rot_1[i] = scipy.ndimage.interpolation.rotate(
                image3_1, angles[2], mode='nearest', axes=(1, 2), reshape=False)
            batch_rot_2[i] = scipy.ndimage.interpolation.rotate(
                image3_2, angles[2], mode='nearest', axes=(1, 2), reshape=False)

        # else:
        #     batch_rot_1[i] = batch_1[i]
        #     batch_rot_2[i] = batch_2[i]
    return batch_rot_1.reshape(size), batch_rot_2.reshape(size)

def random_shift_3d(batch, max_shift):
    size = batch[0].shape
    batch_1, batch_2 = batch
    shifts = [random.uniform(-max_shift, max_shift) for i in range(3)]
    batch_shift_1 = np.zeros(batch_1.shape)
    batch_shift_2 = np.zeros(batch_2.shape)
    for i in range(batch_1.shape[0]):
        batch_shift_1[i] = shift(batch_1[i], shifts, cval=0)
        batch_shift_2[i] = shift(batch_2[i], shifts, cval=0)
    return batch_shift_1.reshape(size), batch_shift_2.reshape(size)


atoms_info = {
    'C': 0,
    'O': 1,
    'N': 2,
    'S': 3,
    'P': 4,
    'CL': 5
}

rev_atoms_info = {
    0: 'C',
    1: 'O',
    2: 'N',
    3: 'S',
    4: 'P',
    5: 'CL'
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
        cmd.iterate_state(
            1, 'all', 'mol.append([x, y, z, elem])', space=locals(), atomic=0)
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
    i = int((p[0] + X*cell_size)/cell_size)
    j = int((p[1] + Y*cell_size)/cell_size)
    k = int((p[2] + Z*cell_size)/cell_size)
    atom_index = int(p[3])

    array[atom_index, int(i/2), int(j/2), int(k/2)] = [(p[0] + X*cell_size)-(2*int(i/2)+1)*cell_size,
                                  (p[1] + Y*cell_size)-(2*int(j/2)+1)*cell_size, 
                                  (p[2] + Z*cell_size)-(2*int(k/2)+1)*cell_size]


def fill_cell(array, p, cell_size=1):
    _, X, Y, Z = array.shape
    i = int((p[0] + X*cell_size/2)/cell_size)
    j = int((p[1] + Y*cell_size/2)/cell_size)
    k = int((p[2] + Z*cell_size/2)/cell_size)
    atom_index = int(p[3])

    array[atom_index, i, j, k] = 1
    
from scipy import signal
def filtering(voxel):
    new_voxel = np.zeros(voxel.shape)
    sigma = 1.0 
    x = np.arange(-1,1,1)
    y = np.arange(-1,1,1)
    z = np.arange(-1,1,1)
    xx, yy, zz = np.meshgrid(x,y,z)
    kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
    # kernel[2,2,2] = kernel[2,2,2]*2
    for i in range(3):
        new_voxel[i] = signal.convolve(voxel[i], kernel, mode="same")
    return new_voxel

def recovery(voxel):
    new_voxel = np.zeros(voxel.shape)
    """
    kernel = np.array([[[0, -1, -1, -1, 0],
    [-1, 2, -4, 2, -1],
    [-1, -4, 13, -4, 6],
    [-1, 2, -4, 2, -1],
    [0, -1, -1, -1, 0]]])
    """
    kernel = np.array([[[1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, -476, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]]])
    kernel = -kernel/256
    # kernel = np.array([[[0,-1,0],[-1,5,-1],[0,-1,0]]])
    for i in range(3):
        new_voxel[i] = signal.convolve(voxel[i], kernel, mode="same")
    return new_voxel

def to_voxel(points, n_cell=20, cell_size=1):
    voxel = np.zeros(n_cell**3 * len(atoms_info), dtype=np.int8)\
        .reshape([len(atoms_info), n_cell, n_cell, n_cell])
    for point in points:
        fill_cell(voxel, point, cell_size)
    return voxel


def to_delta_xyz(points, n_cell=20, cell_size=1):
    voxel = np.zeros(int(n_cell/2)**3 * len(atoms_info) * 3, dtype=np.float)\
        .reshape([len(atoms_info), int(n_cell/2), int(n_cell/2), int(n_cell/2), 3])
    for point in points:
        xyz(voxel, point, cell_size)
    return voxel


def process_grid(array, threshold=[0.2, 0.2, 0.2]):
    new_array = np.zeros(array.shape, dtype=bool)
    for i in range(array.shape[0]):
        for x in range(array.shape[1]):
            for y in range(array.shape[2]):
                for z in range(array.shape[3]):
                    new_array[i, x, y, z] = True if array[i,
                                                          x, y, z] > threshold[i] else False
    return new_array


# %matplotlib inline
def savegrid(d, filename, threshold=[0.2, 0.2, 0.2]):
    # prepare some coordinates
    d = process_grid(d, threshold=threshold)

    # combine the objects into a single boolean array
    voxels = d[0] 
    for i in range(d.shape[0]-1):
        voxels = voxels | d[i+1]

    # set the colors of each object
    colors = np.empty(voxels.shape, dtype=object)
    colors[d[0]] = 'gray'
    colors[d[1]] = 'red'
    colors[d[2]] = 'blue'
    # colors[d[3]] = 'green'
    # colors[d[4]] = 'pink'
    # colors[d[5]] = 'yellow'

    # and plot everything
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
    plt.savefig(filename, dpi=140)

def get_threshold(true_grid, target_grid):
    threshold=[0.5, 0.5, 0.5]
    max_score = 0
    for i in range(3):
        for j in range(20):
            new_threshold = []
            new_threshold[:] = threshold
            new_threshold[i] = j * 0.5/20
            target = [process_grid(target_grid[k], grid_size=20, threshold=new_threshold).reshape(3, -1) for k in range(len(target_grid))] 
        
            target_ = np.array([1 if target[n][l, k] else 0 for k in range(20*20*20) for l in range(3) for n in range(len(target_grid))]).reshape(len(target_grid), 3, -1)
            if max_score < score_grid(true_grid, target_):
                max_score = score_grid(true_grid, target_)
                threshold = new_threshold
            
    return new_threshold
from sklearn.metrics import average_precision_score

def score_grid(true_grid, target_grid, is_rotation=False):
    score = 0
    for i in range(3):
        score += average_precision_score(true_grid[:,i].reshape(true_grid.shape[0], -1)[:,:10],target_grid[:,i].reshape(len(true_grid), -1)[:,:10])
    return score / 3


def Rx(theta):
    return np.matrix([[1, 0, 0],
                     [0, m.cos(theta), -m.sin(theta)],
                     [0, m.sin(theta), m.cos(theta)]])


def Ry(theta):
    return np.matrix([[m.cos(theta), 0, m.sin(theta)],
                     [0, 1, 0],
                     [-m.sin(theta), 0, m.cos(theta)]])


def Rz(theta):
    return np.matrix([[m.cos(theta), -m.sin(theta), 0],
                     [m.sin(theta), m.cos(theta), 0],
                     [0, 0, 1]])


def rotate(points, phi, theta, psi):
    ps, atoms = shaping(points)
    R = Rz(psi) * Ry(theta) * Rz(phi)
    outputs = []
    for i in range(len(ps)):
        t = np.array(R * ps[i]).reshape(1, -1).tolist()[0]
        t.append(atoms[i])
        outputs.append(t)
    return outputs

def to_xyz_file(atoms, filename="test.xyz"):
    with open(filename, "w") as f:
        f.write(str(len(atoms))+"\n\n")
        for a in atoms:
            f.write(rev_atoms_info[a[3]]+"\t"+str(a[0])+"\t"+str(a[1])+"\t"+str(a[2])+"\n")

def to_xyz(atoms, cell_size=20, n_cell=1):
    for i in range(int(n_cell/2)):
        for j in range(int(n_cell/2)):
            for k in range(int(n_cell/2)):
                atoms[i,j,k] = atoms[i,j,k]+[cell_size*2*i+cell_size-int(n_cell/2)*cell_size,
                cell_size*2*j+cell_size-int(n_cell/2)*cell_size,
                cell_size*2*k+cell_size-int(n_cell/2)*cell_size]
import itertools
def gather(voxel):
    new_voxel = np.zeros(voxel.shape)
    for i in range(3):
        for x in range(voxel.shape[1]-1):
            for y in range(voxel.shape[2]-1):
                for z in range(voxel.shape[3]-1):
                    s = [voxel[i, x+dx, y+dy, z+dz] for dx, dy, dz in itertools.product([-1,0,1], repeat=3) if abs(dx)+abs(dy)+abs(dz)!=0]
                    s.sort(reverse=True)
                    if voxel[i, x, y, z] >= s[2] and i == 0:
                        new_voxel[i, x, y, z] = voxel[i, x, y, z]
                    if voxel[i, x, y, z] >= s[0] and i != 0:
                        new_voxel[i, x, y, z] = voxel[i, x, y, z]
    return new_voxel

def voxel_to_xyz(voxel, cell_size=0.5, threshold=[0.9, 0.9, 0.9]):
    points = []
    _, X, Y, Z = voxel.shape
    for i in range(3):
        for x in range(voxel.shape[1]-1):
            for y in range(voxel.shape[2]-1):
                for z in range(voxel.shape[3]-1):
                    if voxel[i, x, y, z] >= threshold[i]:
                        p_x = x*cell_size - X*cell_size/2
                        p_y = y*cell_size - Y*cell_size/2
                        p_z = z*cell_size - Z*cell_size/2
                        points.append([p_x, p_y, p_z, i])
    return points