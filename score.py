from copy import Error
from logging import error
import re
from vina import Vina
import argparse
from utils.preprocess import *
import pymol
import os
import numpy as np
import signal
import atexit
import time
from threading import Thread
from multiprocessing import Process
import multiprocessing
import os
def calc_vina_score(ligand_filename, receptor_filename, center): 

    def run_proc(procnum, value):
        v = Vina(sf_name='vina')
        v.set_receptor(receptor_filename)
        v.set_ligand_from_file(ligand_filename)
        v.compute_vina_maps(center=center, box_size=[32, 32, 32])
        energy = v.score()
        value[procnum]=energy[0]
    def wait():
        pass
        
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    p = Process(target=run_proc, args=(0, return_dict))
    p2 = Process(target=wait)
    p2.start()
    p.start()
    p.join()
    p2.join()
    if len(return_dict.values()) == 0:
        return 999
    else:
        return return_dict.values()[0]

def main(ligand_filename, receptor_filename):
    # v = Vina(sf_name='vina')
    # v.set_receptor(receptor_filename)
    # v.set_ligand_from_file(ligand_filename)

    label = os.path.splitext(ligand_filename)[0].split("/")[-1]
    cmd = pymol.cmd
    mol = []
    cmd.delete("all")
    cmd.load(ligand_filename)
    cmd.h_add(label)
    cmd.iterate_state(1, label, 'mol.append([x, y, z, elem])', space=locals(), atomic=0)
    center=calc_center_ligand(mol)
    # v.compute_vina_maps(center=center, box_size=[32, 32, 32])

    # # Score the current pose
    # energy = v.score()
    energy=calc_vina_score(ligand_filename, receptor_filename, center)
    print('Score before minimization: %.3f (kcal/mol)' % energy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare Protein for AutoDock Vina')
    parser.add_argument('ligand_file')
    parser.add_argument('receptor_file')
    args = parser.parse_args()
    main(args.ligand_file, args.receptor_file)