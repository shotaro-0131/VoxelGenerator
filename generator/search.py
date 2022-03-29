"""
reference 
    https://github.com/pbsinclair42/MCTS
"""
from __future__ import division

from numpy.lib.npyio import savez_compressed
from models.u_net import UNet
from myMcts import MyMcts as mcts
# from mcts import mcts
import hydra
# from icecream import ic
from rdkit import rdBase
from rdkit.Chem import PandasTools, QED
from utils.preprocess import *
from utils import sascorer
from score import calc_vina_score
from os.path import exists

from openbabel import openbabel

import numpy as np
import itertools

from copy import deepcopy
import math

import hashlib
import subprocess
from utils.preprocess import *
import gc
from os.path import exists
from predict import *
import os


class StateInterface():
    def getCurrentPlayer(self):
        # 1 for maximiser, -1 for minimiser
        raise NotImplementedError()

    def getPossibleActions(self):
        raise NotImplementedError()

    def takeAction(self, action):
        raise NotImplementedError()

    def isTerminal(self):
        raise NotImplementedError()

    def getReward(self):
        # only needed for terminal states
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()


class ActionInterface():
    def __eq__(self, other):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError()


class InitAtomAdd(ActionInterface):
    def __init__(self, position, atom_type):
        self.position = position
        self.atom_type = atom_type

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.position == other.position and self.atom_type == other.atom_type

    def __hash__(self):
        return hash((self.position[0], self.position[1], self.position[2], self.atom_type, self.__class__))


class AtomAdd(ActionInterface):
    def __init__(self, position, atom_type, selected_index, bond, connected_index=None, connected_bond=None):
        self.position = position
        self.selected_index = selected_index
        self.atom_type = atom_type
        self.bond = bond

        self.connected_index = connected_index
        self.connected_bond = connected_bond

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.position == other.position and self.atom_type == other.atom_type and self.bond == other.bond

    def __hash__(self):
        return hash((self.position[0], self.position[1], self.position[2], self.atom_type, self.bond, self.__class__, self.connected_index, self.connected_bond))


class BondType:
    def __init__(self, atoms, min_len, max_len, hands):
        self.atoms = atoms
        # voxel base
        # self.min_len = (length-0.25*math.sqrt(3))
        # self.max_len = (length+0.25*math.sqrt(3))

        # openbabel base
        self.min_len = min_len
        self.max_len = max_len
        self.hands = hands
        self.available_voxels = [(dx, dy, dz) for dx, dy, dz in itertools.product(np.arange(-1*2, 2+1, 1), repeat=3) if 0.25*dx*dx +
                                 0.25*dy*dy + 0.25*dz*dz >= self.min_len*self.min_len and 0.25*dx*dx + 0.25*dy*dy + 0.25*dz*dz <= self.max_len*self.max_len]
        self.near_voxels = [(dx, dy, dz) for dx, dy, dz in itertools.product(
            np.arange(-1*2, 2+1, 1), repeat=3) if 0.25*dx*dx + 0.25*dy*dy + 0.25*dz*dz < self.min_len*self.min_len]

    def get(self, atom):
        if atom == self.atoms[0]:
            return self.atoms[1]
        else:
            return self.atoms[0]

    def isin(self, dist):
        return self.min_len <= dist and self.max_len >= dist

    def isOut(self, dist):
        return self.max_len < dist


"""
https://cccbdb.nist.gov/expbondlengths1x.asp
"""
C1 = 0.75
C2 = 0.67
N1 = 0.71
N2 = 0.60
O1 = 0.63
O2 = 0.57
MARGIN = 0.45
CC1 = BondType([0, 0], C2+C1, C1+C1+MARGIN, 1)
CC2 = BondType([0, 0], C2+C2-MARGIN, C2+C1, 2)
# CC3 = BondType([0,0], 1.20-0.45, 1.20, 3)
CO1 = BondType([0, 1], (C2+O2+C1+O1)/2, C1+O1+MARGIN, 1)
CO2 = BondType([0, 1], C2+O2-MARGIN, (C2+O2+C1+O1)/2, 2)
CN1 = BondType([0, 2], (C2+N2+C1+N1)/2, C1+N1+MARGIN, 1)
CN2 = BondType([0, 2], C2+N2-MARGIN, (C2+N2+C1+N1)/2, 2)
# CN3 = BondType([0,2], 1.16-0.45, 1.16, 3)

MAX_LENGTH = [C1+C1+MARGIN, C1+O1+MARGIN, C1+N1+MARGIN]
# BondTypes=[CC1, CC2, CC3, CO1, CO2, CN1, CN2, CN3]
BondTypes = [CC1, CC2, CO1, CO2, CN1, CN2]


class GridState(StateInterface):
    def __init__(self, cfg):
        self.target = cfg.target
        self.use_knowledge = cfg.setting.use_knowledge
        self.voxel = np.load(os.path.join(
            hydra.utils.to_absolute_path(""), cfg.data_dir, cfg.target, cfg.predicted_file))[:3]
        self.raw_voxel = np.load(os.path.join(
            hydra.utils.to_absolute_path(""), cfg.data_dir, cfg.target, cfg.predicted_file))[:3]
        self.data_dir = cfg.data_dir
        self.next_atom = {0: [0, 1, 2], 1: [0], 2: [0]}
        self.atom_hands = {0: 4, 1: 3, 2: 4}
        self.center = calc_center_ligand(get_mol(os.path.join(
            hydra.utils.to_absolute_path(""), cfg.data_dir, cfg.target, cfg.ligand_file)))
        self.grid_size = cfg.setting.grid_size
        self.cell_size = cfg.setting.cell_size

        self.voxel = self.standard(self.voxel)
        self.state_voxel = np.zeros(
            (3, cfg.setting.grid_size, cfg.setting.grid_size, cfg.setting.grid_size))
        self.state_index = []
        self.have_bond = []

        self.bondTypes = [[x for x in BondTypes if t in x.atoms]
                          for t in [0, 1, 2]]
        self.disable_index = []

        self.invalid = [[(dx, dy, dz) for dx, dy, dz in itertools.product(
            np.arange(-1*2, 2+1, 1), repeat=3) if 0.25*(dx*dx + dy*dy + dz*dz) < l*l] for l in MAX_LENGTH]
        self.connected_atoms = [[]]
        self.qed_list = [0]
        self.qed_penalty = cfg.penalty.qed
        self.vina_score_penalty = cfg.penalty.vina_score
        self.qed = cfg.penalty.qed
        self.vina_score = cfg.penalty.vina_score
        self.stop_length = cfg.stop_length
        self.next_actions = []
        self.next_probs = []

    def detectContactAtoms(self, target, t, x, y, z):
        contact_atoms = {}
        # 追加する原子が既にある原子と矛盾しないかをチェック
        for index, (atom, bond) in enumerate(zip(self.state_index, self.have_bond)):
            if index == target:
                continue
            t2, x2, y2, z2 = atom
            dist = math.sqrt((x-x2)*(x-x2)+(y-y2)*(y-y2)+(z-z2)*(z-z2))/2
            if dist < MAX_LENGTH[t2] and dist < MAX_LENGTH[t]:
                if not t in self.next_atom[t2]:
                    return None
            for bondType in self.bondTypes[t2]:
                if bondType.get(t2) != t:
                    continue
                if bondType.isOut(dist):
                    continue
                if bondType.isin(dist):
                    contact_atoms[index] = [bondType, bond]
                else:
                    return None

        return contact_atoms

    def getPossibleActions(self):

        if len(self.next_actions) != 0:
            return self.next_actions

        possibleActions = []
        next_probs = []
        self.sumProb = 0
        self.next_probs = []
        if len(self.state_index) == 0:

            for x in range(self.grid_size//2-2, self.grid_size//2+2):
                for y in range(self.grid_size//2-2, self.grid_size//2+2):
                    for z in range(self.grid_size//2-2, self.grid_size//2+2):
                        possibleActions.append(InitAtomAdd([x, y, z], 0))
                        self.sumProb += self.raw_voxel[0, x, y, z]
                        next_probs.append(self.raw_voxel[0, x, y, z])
            self.next_actions = possibleActions
            self.next_probs = next_probs
            return possibleActions

        for target, state in enumerate(self.state_index):
            t, x, y, z = state

            for bondType in self.bondTypes[t]:
                if bondType.hands + self.have_bond[target] > self.atom_hands[t]-1:
                    continue

                for dx, dy, dz in bondType.available_voxels:
                    new_x = dx + x
                    new_y = dy + y
                    new_z = dz + z
                    new_t = bondType.get(t)
                    if (new_x, new_y, new_z) in self.disable_index:
                        continue

                    # 他の原子との接触判定
                    contact_atoms = self.detectContactAtoms(
                        target, new_t, new_x, new_y, new_z)
                    if contact_atoms == None or len(contact_atoms.keys()) > 1:
                        continue

                    contact_index = None
                    contact_hands = None
                    if len(contact_atoms.keys()) == 1:
                        contact_index = list(contact_atoms.keys())[0]
                        contact_bond, contact_have_atoms = contact_atoms[contact_index]
                        contact_hands = contact_bond.hands
                        if contact_bond.hands + contact_have_atoms > self.atom_hands[contact_bond.get(t)]-1:
                            continue
                        if contact_bond.hands + bondType.hands > self.atom_hands[new_t]-1:
                            continue
                    if new_x >= 0 and new_y >= 0 and new_z >= 0 and new_x < self.grid_size and new_y < self.grid_size and new_z < self.grid_size:
                        if not self.use_knowledge or self.voxel[bondType.get(t), new_x, new_y, new_z] > 0:
                            possibleActions.append(AtomAdd((new_x, new_y, new_z), bondType.get(
                                t), target, bondType.hands, contact_index, contact_hands))
                            self.sumProb += self.raw_voxel[new_t,
                                                           new_x, new_y, new_z]
                            next_probs.append(
                                self.raw_voxel[new_t, new_x, new_y, new_z])
        self.next_actions = possibleActions
        self.next_probs = next_probs
        return possibleActions

    def min_max(self, voxel):
        for i in range(voxel.shape[0]):
            max_num = np.max(voxel[i])
            voxel[i] = voxel[i]/max_num
        return voxel

    def mol_check(self):
        return 0

    def index2point(self):
        points = []
        def position(x): return x*self.cell_size - \
            0.5*self.cell_size*self.grid_size
        for t, x, y, z in self.state_index:
            points.append([position(x), position(y), position(z), t])
        return points

    def standard(self, voxel, axis=None):
        if axis == None:
            voxel = (voxel-np.mean(voxel.flatten()))/np.std(voxel.flatten())
            return voxel
        for i in range(voxel.shape[0]):
            voxel[i] = (voxel[i] - np.mean(voxel[i].flatten())) / \
                np.std(voxel[i].flatten())
        return voxel

    def findUnvisitedChild(self):
        return [c for c in self.children if not c.isVisited]

    def takeAction(self, action):
        if action.__class__ == InitAtomAdd(None, None).__class__:
            newState = deepcopy(self)
            newState.state_voxel[action.atom_type, action.position[0],
                                 action.position[1], action.position[2]] = 1
            newState.state_index.append(
                [action.atom_type, action.position[0], action.position[1], action.position[2]])
            newState.next_actions = []
            newState.have_bond.append(0)

            for s in newState.state_index:
                t, x, y, z = s
                newState.state_voxel[t, x, y, z] = 1

            return newState

        newState = deepcopy(self)
        newState.state_voxel[action.atom_type, action.position[0],
                             action.position[1], action.position[2]] = 1
        newState.state_index.append(
            [action.atom_type, action.position[0], action.position[1], action.position[2]])
        newBond = action.bond
        disable_index = []
        if action.connected_index != None:
            newState.have_bond[action.connected_index] = self.have_bond[action.connected_index] + \
                action.connected_bond
            newBond = newBond + action.connected_bond
            if self.have_bond[action.connected_index] + action.connected_bond >= self.atom_hands[self.state_index[action.connected_index][0]]-1:
                c_pos = newState.state_index[action.connected_index]
                disable_index.extend([(c_pos[1]+dx, c_pos[2]+dy, c_pos[3]+dz) for dx, dy, dz in self.invalid[c_pos[0]] if c_pos[1]+dx >= 0 and c_pos[2] +
                                     dy >= 0 and c_pos[3]+dz >= 0 and c_pos[1]+dx < self.grid_size and c_pos[2]+dy < self.grid_size and c_pos[3]+dz < self.grid_size])
            if action.connected_bond+action.bond >= self.atom_hands[action.atom_type]-1:
                disable_index.extend([(action.position[0]+dx, action.position[1]+dy, action.position[2]+dz) for dx, dy, dz in self.invalid[action.atom_type] if action.position[0]+dx >= 0 and action.position[1] +
                                     dy >= 0 and action.position[2]+dz >= 0 and action.position[0]+dx < self.grid_size and action.position[1]+dy < self.grid_size and action.position[2]+dz < self.grid_size])

        if action.bond >= self.atom_hands[action.atom_type]-1:
            disable_index.extend([(action.position[0]+dx, action.position[1]+dy, action.position[2]+dz) for dx, dy, dz in self.invalid[action.atom_type] if action.position[0]+dx >= 0 and action.position[1] +
                                 dy >= 0 and action.position[2]+dz >= 0 and action.position[0]+dx < self.grid_size and action.position[1]+dy < self.grid_size and action.position[2]+dz < self.grid_size])

        if action.bond+self.have_bond[action.selected_index] >= self.atom_hands[newState.state_index[action.selected_index][0]]-1:
            s_pos = newState.state_index[action.selected_index]
            disable_index.extend([(s_pos[1]+dx, s_pos[2]+dy, s_pos[3]+dz) for dx, dy, dz in self.invalid[s_pos[0]] if s_pos[1]+dx >= 0 and s_pos[2] +
                                 dy >= 0 and s_pos[3]+dz >= 0 and s_pos[1]+dx < self.grid_size and s_pos[2]+dy < self.grid_size and s_pos[3]+dz < self.grid_size])

        newState.disable_index.extend(disable_index)
        newState.disable_index = list(set(newState.disable_index))
        newState.have_bond[action.selected_index] = newState.have_bond[action.selected_index] + action.bond
        newState.have_bond.append(newBond)
        newState.next_actions = []

        for s in newState.state_index:
            t, x, y, z = s
            newState.state_voxel[t, x, y, z] = 1
        return newState

    def isTerminal(self):

        if len(self.state_index) >= self.stop_length:
            qed, sa_score, vina_score = self.get_scores()
            with open(f"{self.target}.csv", "a") as file_object:
                file_object.write(
                    f"{self.__hash__()}, {qed}, {sa_score}, {vina_score}, {np.sum(self.raw_voxel*self.state_voxel)}, {len(self.state_index)}\n")
            return True

        if len(self.getPossibleActions()) == 0:
            qed, sa_score, vina_score = self.get_scores()
            with open(f"{self.target}.csv", "a") as file_object:
                file_object.write(
                    f"{self.__hash__()}, {qed}, {sa_score}, {vina_score}, {np.sum(self.raw_voxel*self.state_voxel)}, {len(self.state_index)}\n")
            return True
        return False

    def getReward(self):

        if len(self.state_index) > 10:
            return -self.vina_score
        else:
            return -self.vina_score_penalty

    def origin(self, atoms, c):
        new_atoms = []
        for a in atoms:
            b = np.array(a[:3] + c).tolist()
            b.append(a[3])
            new_atoms.append(b)
        return new_atoms

    def get_scores(self):
        warnings.filterwarnings('ignore')
        points = self.index2point()

        to_xyz_file(self.origin(points, self.center),
                    f"{self.target}/mols/tmp{len(self.state_index)}.xyz")
        obConversion1 = openbabel.OBConversion()
        obConversion2 = openbabel.OBConversion()
        mol = openbabel.OBMol()
        obConversion1.SetInAndOutFormats("xyz", "sdf")
        obConversion2.SetInAndOutFormats("sdf", "pdbqt")
        obConversion1.ReadFile(
            mol, f"{self.target}/mols/tmp{len(self.state_index)}.xyz")

        obConversion1.WriteFile(
            mol, f"{self.target}/mols/{self.__hash__()}.sdf")
        cmd = pymol.cmd

        cmd.delete("all")
        cmd.load(f"{self.target}/mols/{self.__hash__()}.sdf")
        cmd.h_add("all")
        cmd.save(f"{self.target}/mols/{self.__hash__()}.sdf")
        obConversion2.ReadFile(
            mol, f"{self.target}/mols/{self.__hash__()}.sdf")
        obConversion2.WriteFile(
            mol, f"{self.target}/mols/{self.__hash__()}.pdbqt")

        gc.collect()
        try:
            df = PandasTools.LoadSDF(
                f"{self.target}/mols/{self.__hash__()}.sdf")
            df["QED"] = df.ROMol.map(QED.qed)
            df['SA_score'] = df.ROMol.map(sascorer.calculateScore)
            if not exists(f"{self.target}/mols/{self.__hash__()}.pdbqt"):
                vina_score = self.vina_score_penalty
            else:
                vina_score = calc_vina_score(f"{self.target}/mols/{self.__hash__()}.pdbqt", os.path.join(
                    hydra.utils.to_absolute_path(f"{self.data_dir}/{self.target}/receptor.pdbqt")), self.center)

            self.qed = df["QED"][0]
            self.vina_score = vina_score
            return df["QED"][0], df["SA_score"][0], vina_score
        except Exception as e:
            self.qed = self.qed_penalty
            self.vina_score = self.vina_score_penalty
            return self.qed_penalty, 10, self.vina_score_penalty

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.state_index == other.state_index

    def __hash__(self, state_index=None):
        if state_index == None:
            return hashlib.md5(str(hash((self.__class__, str(self.state_index)))).encode()).hexdigest()
        else:
            return hashlib.md5(str(hash((self.__class__, str(state_index)))).encode()).hexdigest()


@hydra.main(config_name="search_params.yml")
def main(cfg):
    assert cfg.target != ""

    if not exists(f"{cfg.target}"):
        os.mkdir(f"{cfg.target}")

    if not exists(f"{cfg.target}/mols"):
        os.mkdir(f"{cfg.target}/mols")

    state = GridState(cfg)
    searcher = mcts(iterationLimit=cfg.iterationLimit,
                    use_knowledge=cfg.setting.use_knowledge)

    with open(f"{cfg.target}.csv", "w") as file_object:
        file_object.write("hash, qed, sascore, vina_socre, score, length\n")

    searcher.search(initialState=state)


if __name__ == "__main__":
    main()
