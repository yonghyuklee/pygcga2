#!/usr/bin/env python
import re
from collections.abc import Iterable
import numpy as np
from ase import Atoms
from ase.build import molecule as ase_create_molecule
from ase.data import covalent_radii, chemical_symbols
from pygcga2.topology import Topology
from pygcga2.checkatoms import CheckAtoms
import sys, os
from pygcga2.utilities import NoReasonableStructureFound
from ase.atom import Atom
from ase.constraints import FixAtoms
from ase.constraints import Hookean
from ase.constraints import FixBondLengths
from ase.constraints import FixedLine
try:
    from ase.constraints import NeverDelete
except ImportError:
    NeverDelete = type(None)
from ase.neighborlist import NeighborList
from ase.neighborlist import neighbor_list
from ase.neighborlist import natural_cutoffs
import random
import numpy as np
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import *
from ase.io.trajectory import TrajectoryWriter
import json
import random
import itertools
from ase.ga.utilities import (atoms_too_close,
                              atoms_too_close_two_sets,
                              gather_atoms_by_tag,
                              get_rotation_matrix, get_all_atom_types, closest_distances_generator)
from math import cos, sin, pi
import numpy.ma as ma
import random

BOND_LENGTHS = dict(zip(chemical_symbols, covalent_radii))
debug_verbosity = True

def selecting_index(weights=None):
    """This function accept a list of float as input, all the inputs (weights) are non-negative number.
    Then, this function choose on number by the weight specified"""
    if weights is None:
        raise RuntimeError("weights is None?")
    _sumofweights = np.asarray(weights).sum()
    if _sumofweights < 1.0e-6:
        raise NoReasonableStructureFound("Sum of weights are too small, I can not find a good site")
    _w = np.array(weights)/_sumofweights
    r = np.random.uniform(low=0.0, high=1.0)
    for i in range(len(_w)):
        if i == 0 and r <= _w[0]:
            return 0
        elif sum(_w[:i+1]) >= r > sum(_w[:i]):
            return i
    return len(_w) - 1

def generate_H_site(r0, distance=None, width=None, center_of_mass=None):
    """
    r0 :
        is the position of Pt atom, the equilibrium distance of H and Pt is 1.36+0.70=2.06
    center_of_mass:
        it is the center of the cluster, if the center_of_mass is specified, the position of new atom will be placed
        on the outer direction from center_of_mass
    """
    if distance is None:
        distance = BOND_LENGTHS['Pt'] + BOND_LENGTHS['H']
    if width is None:
        width = distance*0.15
    dr = np.random.normal(loc=distance, scale=width)
    if center_of_mass is None:
        theta = np.arccos(1.0-2.0*np.random.uniform(low=0.0, high=1.0))
    else:
        # new H atoms only exist in the z+ direction
        theta = np.arccos((1.0-np.random.uniform(low=0.0, high=1.0)))
    phi = np.random.uniform(low=0.0, high=np.pi * 2.0)
    dz = dr * np.cos(theta)
    dx = dr * np.sin(theta) * np.cos(phi)
    dy = dr * np.sin(theta) * np.sin(phi)
    dp0 = np.array([dx, dy, dz])
    try:
        assert np.abs(np.sqrt(np.power(dp0, 2).sum()) - dr) < 1.0e-3 # check math is correct
    except AssertionError:
        print("Whoops, Bad things happened when generate new H positions")
        exit(1)
    if center_of_mass is None:
        return dp0 + np.array(r0)  # returned value is a possible site for H atom
    else:
        pout = r0 - np.array(center_of_mass)
        x, y, z = pout
        # we will find a rotation matrix to rotate the z+ direction to pout here.
        if np.power(pout, 2).sum() < 1.0e-1:
            # r0 is too close to the center of mass? this is a bad Pt site
            # if the cluster is a small cluster, this might be correct
            return None
        if (y ** 2 + z ** 2) < 1.0e-6:
            theta_y = np.pi / 2.0
            RotationMatrix = np.array([[np.cos(theta_y), 0.0, np.sin(theta_y)],
                                       [0.0, 1.0, 0.0],
                                       [-1.0 * np.sin(theta_y), 0.0, np.cos(theta_y)]])
        else:
            # fint the rotation matrix
            with np.errstate(invalid='raise'):
                # try:
                theta_x = -1.0 * np.arccos(z / np.sqrt(y ** 2 + z ** 2))
                theta_y = np.arcsin(x / np.sqrt(x ** 2 + y ** 2 + z ** 2))
                # except:
                #     print "x= %.8f, y=%.8f, z=%.8f" % (x,y,z)
                #     print z / (y ** 2 + z ** 2)
                #     print x / (x ** 2 + y ** 2 + z ** 2)
                #     print("something erorr in the math here")
                #     exit(1)
            M_x = np.array([[1.0, 0.0, 0.0],
                            [0.0, np.cos(theta_x), -1.0 * np.sin(theta_x)],
                            [0.0, np.sin(theta_x), np.cos(theta_x)]])
            M_y = np.array([[np.cos(theta_y), 0.0, np.sin(theta_y)],
                            [0.0, 1.0, 0.0],
                            [-1.0 * np.sin(theta_y), 0.0, np.cos(theta_y)]])
            RotationMatrix = np.matmul(M_y, M_x)
        # RotationMatrix could rotate the [0,0,1] to pout direction.
        dp1 = np.matmul(RotationMatrix, dp0)
        try:
            # check math is correct
            assert np.abs(np.power(dp1, 2).sum() - np.power(dr, 2)) < 1.0e-3
        except AssertionError:
            _r2 = np.power(dp1, 2).sum()
            print("_r2=%.3f" % _r2)
            print("r**2=%.3f" % dr ** 2)
            exit(1)
        return dp1 + np.array(r0)


def add_molecule_on_cluster(cluster, molecule='H', anchor_atom=None, metal="Pt",
                            verbosity=False, maxcoordination=12, weights=None,
                            max_attempts_1=500,
                            max_attempts_2=500,
                            minimum_bond_distance_ratio=0.7,
                            maximum_bond_distance_ratio=1.4,
                            outer_sphere=True,
                            distribution=0):
    """
    :param cluster: ase.atoms.Atoms object
    :param molecule: string. I should be able get the molecule from ase.build.molecule
    :param anchor_atom: string if there are more than 2 atoms in the molecule, which atom will form bond with metal,
           This parameter currently is not used.
    :param metal: string; metal element; currently on support one element cluster (no alloy allowed)
    :param maxcoordination: the maximum coordination for 'metal',the anchoring is attempted only if CN of meta
            is smaller than maxcoordition number
    :param weights: list of floats. If the weights are specified, I will not check the coordination numbers, the
           anchoring sites will be determined by the weights.
           The length of weights must be same as the number of atoms in the 'cluster' object
    :param verbosity: print details to standard out
    :param max_attempts_1: how many attempts made to find a suitable anchor site on cluster
    :param max_attempts_2: how many attempts made to anchor the molecule on the cluster
    :param minimum_bond_distance_ratio: The minimum bond distance will calculated by this ratio multiplied by the
           sum of atomic radius (got from ase.data)
    :param maximum_bond_distance_ratio: bla,bla
    :param outer_sphere: if True, the molecule will be added in the outer sphere region of the cluster, this is good
    option for large and spherical clusters (center of the mass is used). Other wise, use False, for small cluster
    :param distribution: 0 or 1: if distribution=0, the maximum coordination number of metal
    is maxcoordination; if distribution=1, the probability is uniformly distributed for different metal atoms.
    :return: ase.atoms.Atoms object
    This function will add one molecule on the cluster and return a new cluster.
    ++ using of weights ++
    If weights are not specified, the anchoring sites will be determined by the coordination number of the metal atoms,
    otherwise, the anchoring sites will be chosen by the weights. For example, the weights could be determined by atomic
    energies from high-dimensional neural network.
    ++ adding molecule rather than single atom++
    Currently this function does not support add a molecule (more than 2 atoms), so the anchor_atom is ignored.
    ++ the total attempts will be max_attempts_1 x max_attempts_2
    ++ minimum_bond_distance
    """
    if cluster is None:
        raise RuntimeError("adding molecule on the None object")
    try:
        _mole = ase_create_molecule(molecule)
        if _mole.get_number_of_atoms() == 1:
            anchor_atom = molecule.strip()
        elif anchor_atom is None:
            raise RuntimeError("You must set anchor_atom when a molecule (natoms > 1) is adsorbed")
        else:
            assert anchor_atom in _mole.get_chemical_symbols()
    except KeyError as e:
        raise RuntimeError("I can not find the molecule {}".format(e))

    cluster_topology = Topology(cluster, ratio=1.3)

    symbols = cluster.get_chemical_symbols()
    props = [] # The probabilities of attaching _mole to sites
    adsorption_sites=[] # adsorption_sites is a list of elements which can absorb molecule
    if isinstance(metal,str):
        adsorption_sites.append(metal)
    elif isinstance(metal, list):
        for m in metal:
            adsorption_sites.append(m)
    else:
        raise RuntimeError("metal=str or list, for the adsorption sites")

    # setup the probabilities for adsorbing _mole
    for index, symbol in enumerate(symbols):
        if weights is not None:
            props.append(weights[index])
        elif symbol not in adsorption_sites:
            props.append(0.0)
        else:
            if distribution == 0:
                _cn = cluster_topology.get_coordination_number(index)
                props.append(max([maxcoordination-_cn, 0.0]))
            else:
                props.append(1.0)

    atoms_checker=CheckAtoms(max_bond=maximum_bond_distance_ratio, min_bond=minimum_bond_distance_ratio)
    for n_trial in range(max_attempts_1):
        if verbosity:
            sys.stdout.write("Try to find a metal atoms, trial number %3d\n" % n_trial)
        attaching_index = selecting_index(props)
        site_symbol = symbols[attaching_index]
        if site_symbol not in adsorption_sites:
            continue
        else:
            for n_trial_2 in range(max_attempts_2):
                p0 = cluster.get_positions()[attaching_index]
                if outer_sphere:
                    p1 = generate_H_site(p0, distance=BOND_LENGTHS[site_symbol]+BOND_LENGTHS[anchor_atom],
                                         width=0.05, center_of_mass=cluster.get_center_of_mass())
                else:
                    p1 = generate_H_site(p0, distance=BOND_LENGTHS[site_symbol]+BOND_LENGTHS[anchor_atom],
                                         width=0.05, center_of_mass=None)
                if p1 is None:
                    # I did not find a good H position
                    continue
                if _mole.get_number_of_atoms() == 1:
                    _mole.set_positions(_mole.get_positions() - _mole.get_center_of_mass() + p1)
                    new_atoms = cluster.copy()
                    for _a in _mole:
                        _a.tag = 1
                        new_atoms.append(_a)
                    if atoms_checker.is_good(new_atoms, quickanswer=True):
                        return new_atoms
                else:
                    # if _mole has more than 1 atoms. I will try to rotate the _mole several times to give it a chance
                    # to fit other atoms
                    for n_rotation in range(20):
                        new_atoms = cluster.copy()
                        # firstly, randomly rotate the _mole
                        phi = np.rad2deg(np.random.uniform(low=0.0, high=np.pi * 2.0))
                        theta = np.rad2deg(np.arccos(1.0 - 2.0 * np.random.uniform(low=0.0, high=1.0)))
                        _mole.euler_rotate(phi=phi, theta=theta, psi=0.0, center="COP")
                        # get the positions of anchored atoms
                        possible_anchored_sites = [i for i, e in enumerate(_mole.get_chemical_symbols())
                                                   if e == anchor_atom]
                        decided_anchored_site = np.random.choice(possible_anchored_sites)
                        p_anchored_position =_mole.get_positions()[decided_anchored_site]
                        # make sure the anchored atom will locate at p1, which is determined by the function
                        _mole.set_positions(_mole.get_positions()+(p1-p_anchored_position))
                        for _a in _mole:
                            _a.tag = 1
                            new_atoms.append(_a)
                        if atoms_checker.is_good(new_atoms, quickanswer=True):
                            return new_atoms
                        else:
                            continue
    raise NoReasonableStructureFound("No reasonable structure found when adding an addsorbate")

def remove_H(atoms=None, molecule="H", adsorbate_weights=None):
    H_ndx = [atom.index for atom in atoms if atom.symbol=='H']
    sel_ndx = random.choice(H_ndx)
    del atoms[sel_ndx]

    return atoms
    
def delete_single_atom_with_constraint(atoms=None, indice_to_delete=None):
    if indice_to_delete is None:
        return # nothing to delete
    elif indice_to_delete > len(atoms)-1:
        raise RuntimeError("try too remove too many atoms?")

    # I have to teach each fix objects individually, how can I make it smarter?
    hookean_pairs = []
    fix_atoms_indices = []
    fix_bondlengths_indices = None
    never_delete_indices = []
    fix_lines_indices = []
    for c in atoms.constraints:
        if isinstance(c, FixAtoms):
            fix_atoms_indices.append(c.get_indices())
        elif isinstance(c, Hookean):
            hookean_pairs.append([c.get_indices(), c.threshold, c.spring])
        elif isinstance(c, FixBondLengths):
            if fix_bondlengths_indices is not None:
                raise RuntimeError("Do you have more than one FixBondLengths?")
            fix_bondlengths_indices = c.pairs.copy()
        elif isinstance(c, NeverDelete):
            for ind in c.get_indices():
                never_delete_indices.append(ind)
        elif isinstance(c, FixedLine):
            fix_lines_indices.append([c.a, c.dir])
        else:
            try:
                cn = c.__class__.__name__
            except AttributeError:
                cn = "Unknown"
            raise RuntimeError("constraint type %s is not supported" % cn)

    new_constraints = []
    for old_inds in fix_atoms_indices:
        new_inds = []
        for ind in old_inds:
            if ind < indice_to_delete:
                new_inds.append(ind)
            elif ind > indice_to_delete:
                new_inds.append(ind - 1)
            else:
                continue
        if len(new_inds) > 0:
            new_constraints.append(FixAtoms(indices=new_inds))
    for old_inds, rt, k in hookean_pairs:
        if indice_to_delete in old_inds:
            continue
        new_inds = []
        for ind in old_inds:
            if ind < indice_to_delete:
                new_inds.append(ind)
            else:
                new_inds.append(ind-1)
        new_constraints.append(Hookean(a1=new_inds[0], a2=new_inds[1], rt=rt, k=k))
    for ind, direction in fix_lines_indices:
        if ind == indice_to_delete:
            continue
        elif ind < indice_to_delete:
            new_constraints.append(FixedLine(ind, direction=direction))
        else:
            new_constraints.append(FixedLine(ind-1,direction=direction))

    if fix_bondlengths_indices is not None:
        number_of_cons, ncols = fix_bondlengths_indices.shape
        assert ncols == 2
        new_inds = []
        for ic in range(number_of_cons):
            ind1, ind2 = fix_bondlengths_indices[ic][0], fix_bondlengths_indices[ic][1]
            if ind1 == indice_to_delete or ind2 == indice_to_delete:
                continue
            else:
                pairs = []
                if ind1 < indice_to_delete:
                    pairs.append(ind1)
                else:
                    pairs.append(ind1-1)
                if ind2 < indice_to_delete:
                    pairs.append(ind2)
                else:
                    pairs.append(ind2-1)
                new_inds.append(pairs)
        if len(new_inds) > 0:
            new_constraints.append(FixBondLengths(pairs=new_inds))
    if len(never_delete_indices) > 0:
        new_inds = []
        for ind in never_delete_indices:
            if ind < indice_to_delete:
                new_inds.append(ind)
            elif ind > indice_to_delete:
                new_inds.append(ind - 1)
        if len(new_inds) > 0:
            new_constraints.append(NeverDelete(indices=new_inds))

    # remove all the constraints
    atoms.set_constraint()
    del atoms[indice_to_delete]

    # add back the constraints
    atoms.set_constraint(new_constraints)
    return

def checkatoms(atoms, bond_range):
    _d = atoms.get_all_distances(mic=True)
    brange = {}
    for key, value in bond_range.items():
        brange[frozenset(key)] = value
    symbols = atoms.get_chemical_symbols()
    n = atoms.get_global_number_of_atoms()
    for i in range(n):
        for j in range(i + 1, n):
            k = (symbols[i],symbols[j])
            dmin = brange[frozenset(k)]
            if _d[i][j] < dmin:
                return False
    return True

def find_surf(atoms, el='Cu', maxCN=12, minCN=5):
    '''
    find the the index of atoms forming the surface of a slab
    '''
    atoms.pbc = [1,1,0]
    nat_cut = natural_cutoffs(atoms, mult=0.95)
    nl = NeighborList(nat_cut, self_interaction=False, bothways=True)
    nl.update(atoms)

    pos = atoms.get_positions()
    Cu_ndx = [atom.index for atom in atoms if atom.symbol==el]
    pos_Cu = np.take(pos,Cu_ndx,axis=0)
    posz_Cu = pos_Cu[:,2]
    pmid = (np.max(posz_Cu)+np.min(posz_Cu))/2
    Cu_ndx_ = [Cu_ndx[i] for i,p in enumerate(posz_Cu) if p>pmid]

    crd=[]
    for n in Cu_ndx_:
        indices, offsets = nl.get_neighbors(n)
        x = np.intersect1d(indices, Cu_ndx)
        crd.append(len(x))

    print(crd)
    surf_ndx=[]
    for i,c in enumerate(crd):
        if c<=maxCN and c>=minCN:
            surf_ndx.append(Cu_ndx_[i])

    return surf_ndx

def remove_CuCO(surface,element="Cu"):
    def flatten(lis):
        for item in lis:
            if isinstance(item, Iterable) and not isinstance(item, str):
                for x in flatten(item):
                    yield x
            else:
                yield item

    nat_cut = natural_cutoffs(surface, mult=0.85)
    nl = NeighborList(nat_cut, self_interaction=False, bothways=True)
    nl.update(surface)
    pos = surface.get_positions()
    posz = pos[:,2]
    Cu_ndx = [atom.index for atom in surface if atom.symbol==element]
    C_ndx = [atom.index for atom in surface if atom.symbol == 'C']
    O_ndx =[atom.index for atom in surface if atom.symbol == 'O']
    surf_ndx = find_surf(surface, el='Cu', maxCN=3, minCN=0)

    if surf_ndx==[]:
        return surface
    sel_Cu_ndx = np.random.choice(surf_ndx,1)[0]
    indices, offsets = nl.get_neighbors(sel_Cu_ndx)
    sel_C_ndx = list(np.intersect1d(indices, C_ndx))
    sel_O_ndx=[]
    for x in sel_C_ndx:
        indices, offsets = nl.get_neighbors(x)
        sel_O_ndx.append(list(np.intersect1d(indices, O_ndx)))

    del_ndx = [sel_Cu_ndx, sel_C_ndx, sel_O_ndx]
    del_ndx = list(flatten(del_ndx))
    del_ndx = np.sort(del_ndx)[::-1]
    for i in del_ndx:
        del surface[i]
    return surface

def add_CuCO(surface,z,element="Cu", bond_range=None, max_trial=50):
    pos = surface.get_positions()
    posz = pos[:,2]
    Cu_ndx = [atom.index for atom in surface if atom.symbol=='Cu']
    surf_ind = find_surf(surface, el='Cu')
    cell=surface.get_cell()
    for _ in range(max_trial):
        n_choose = np.random.choice(surf_ind)
        x=pos[n_choose,0]+np.random.uniform(-1,1,1)[0]
        y=pos[n_choose,1]+np.random.uniform(-1,1,1)[0]
        t=surface.copy()
        t.append(Atom(symbol=element,position=[x,y,pos[n_choose,2]+z],tag=1))
        t.append(Atom(symbol='C',position=[x,y,pos[n_choose,2]+z+1.85],tag=1))
        t.append(Atom(symbol='O',position=[x,y,pos[n_choose,2]+z+1.175+1.85],tag=1))
        inspect = checkatoms(t, bond_range)
        if inspect:
            return t
    raise NoReasonableStructureFound("No good structure found using randomize")
        
def move_CuCO(surface,z,element="Cu", bond_range=None, max_trial=50):
    '''
    z = height at which another atom needed to be added
    '''
    t = remove_CuCO(surface,element="Cu")
    t2 = add_CuCO(t,z=2.2,element="Cu", bond_range=bond_range, max_trial=max_trial)
    return t2

def add_CO(surface, zmax=1.9, zmin= 1.8, bond_range=None, max_trial=50):
    surf_ind = find_surf(surface,el='Cu', maxCN=12, minCN=6)
    pos = surface.get_positions()
    for _ in range(max_trial):
        n_choose = np.random.choice(surf_ind)
        x=pos[n_choose,0]+np.random.uniform(-1,1,1)[0]
        y=pos[n_choose,1]+np.random.uniform(-1,1,1)[0]
        z=pos[n_choose,2]+np.random.uniform(zmin,zmax,1)[0]
        t=surface.copy()
        t.append(Atom(symbol='C',position=[x,y,z],tag=1))
        t.append(Atom(symbol='O',position=[x,y,z+1.175],tag=1))
        inspect = checkatoms(t, bond_range)
        if inspect:
            return t
    raise NoReasonableStructureFound("No good structure found using randomize")

def add_H(surface, bond_range=None, zmin=1.4, zmax=1.7,max_trial=50):
    surf_ind = find_surf(surface,el='Pt', maxCN=12, minCN=1)
    pos = surface.get_positions()
    for _ in range(max_trial):
        n_choose = np.random.choice(surf_ind)
        x=pos[n_choose,0]+np.random.uniform(-1,1,1)[0]
        y=pos[n_choose,1]+np.random.uniform(-1,1,1)[0]
        z=pos[n_choose,2]+np.random.uniform(zmin,zmax,1)[0]
        t=surface.copy()
        t.append(Atom(symbol='H',position=[x,y,z],tag=1))
        inspect = checkatoms(t, bond_range)
        if inspect:
            return t
    raise NoReasonableStructureFound("No good structure found using randomize")
        
def remove_CO(structures):
    atoms = structures.copy()
    nat_cut = natural_cutoffs(atoms, mult=0.85)
    nl = NeighborList(nat_cut, self_interaction=False, bothways=True)
    nl.update(atoms)
    C_ndx = [atom.index for atom in atoms if atom.symbol=='C']
    O_ndx = [atom.index for atom in atoms if atom.symbol=='O']
    O_ndx_sort=[]
    for i in C_ndx:
        indices, offsets = nl.get_neighbors(i)
        O_ndx_sort.append(np.intersect1d(indices,O_ndx))

    n_choose = np.random.choice(list(np.arange(len(C_ndx))))
    del_ndx = np.sort([C_ndx[n_choose], O_ndx_sort[n_choose]])[::-1]
    for i in del_ndx:
        del atoms[i]

    return atoms

def move_CO(structures, bond_range, max_trial=50):
    t = remove_CO(structures)
    t2 = add_CO(t, bond_range=bond_range, max_trial=max_trial)
    return t2

def randomize(surface, dr, bond_range, max_trial=10):

    c_indx = [atom.index for atom in surface if atom.symbol=='C']
    o_indx = [atom.index for atom in surface if atom.symbol=='O']
    pt_indx = [atom.index for atom in surface if atom.symbol=='Cu']

    cutoff = natural_cutoffs(surface, 1.1)
    nl = NeighborList(cutoff , self_interaction=False,  bothways=True)
    nl.update(surface)

    O_ndx_sort=[]
    for i in c_indx:
        indices, offsets = nl.get_neighbors(i)
        O_ndx_sort.append(np.intersect1d(indices,o_indx))

    for _ in range(max_trial):
        ro = surface.get_positions()
        rn = ro.copy()
        t = surface.copy()

        for i,n in enumerate(c_indx):
            disp = np.random.uniform(-1., 1., [1, 3])
            rn[n,:] = ro[n,:]+dr*disp
            rn[n-1,:] = ro[n-1,:]+dr*disp            
            # rn[O_ndx_sort[i],:] = ro[O_ndx_sort[i],:]+dr*disp

        t.set_positions(rn)
        inspect = checkatoms(t, bond_range)
        if inspect:
            return t        
    raise NoReasonableStructureFound("No good structure found at function add_atom_on_surfaces")    

def randomize_all(surface, dr, bond_range, max_trial=10):
    
    fixed = surface.constraints[0].index

    for _ in range(max_trial):
        ro = surface.get_positions()
        rn = ro.copy()
        t = surface.copy()
        for i,n in enumerate(range(len(surface))):
            if i in fixed:
                continue
            disp = np.random.uniform(-1., 1., [1, 3])
            rn[n,:] = ro[n,:]+dr*disp

        t.set_positions(rn)
        inspect = checkatoms(t, bond_range)
        if inspect:
            return t
        
    raise NoReasonableStructureFound("No good structure found using randomize")

def md_nve(surface, calc, T=500, timestep=1, steps=30):
    t = surface.copy()
    t.set_calculator(calc)
    MaxwellBoltzmannDistribution(t, temperature_K=T, force_temp=True)
    dyn = VelocityVerlet(t, timestep * units.fs)
    dyn.run(steps)
    return t

def nve_n2p2(atoms, z_fix, bond_range, map_atoms=[13,1,78,6,8], line_read=1100):

    f = open('Current_Status.json')
    info = json.load(f)
    nstep = info['nsteps']

    # os.system('$(which mpirun) -n 96 python nve_mod.py '+ str(N))
    
    T = random.randint(500,2000)
    N = random.randint(500,1000)
    atoms = read('Current_atoms.traj')
    n = len(atoms)
    ase_adap = AseAtomsAdaptor()
    atoms_ = ase_adap.get_structure(atoms)
    ld = LammpsData.from_structure(atoms_,atom_style='atomic')
    ld.write_file("struc.data")
    os.system('mpirun -n 96 lmp_mpi -var T '+str(T)+' -var N '+str(N)+' < in.nve')

    images = read('md.lammpstrj',':')
    trajw = TrajectoryWriter('md%05i.traj'% nstep,'a')

    f = open("log.lammps","r")
    Lines = f.readlines()
    patten= r'(\d+\s+\-+\d*\.?\d+)'
    e_pot = []
    for i,line in enumerate(Lines):
        if i<line_read:
            continue
        s = line.strip()
        match= re.match(patten, s)
        if match != None:
            D = np.fromstring(s, sep=' ')
            e_pot.append(D[1])

    f_all=[]
    for atoms in images:
        f = atoms.get_forces()
        f_all.append(f)

    for i,atoms in enumerate(images):
        an = atoms.get_atomic_numbers()

        for k,j in enumerate(map_atoms):
            an = [j if x==k+1 else x for x in an]

        atoms.set_atomic_numbers(an)
        trajw.write(atoms, energy=e_pot[i], forces=f_all[i])

    t = read('md%05i.traj@-1'%nstep)

    if not os.path.isdir('md_files'):
        os.mkdir('md_files')
    
    os.system('mv md%05i.traj md_files/'% nstep)
    os.system('mv log.lammps md_files/')
    #os.system('rm md.lammpstrj')
    # os.system('rm md.traj')

    inspect = checkatoms(t, bond_range)
    if inspect:
        return t

    raise NoReasonableStructureFound("Bond range didn't satisfy in the NVT calculations")

def mirror_mutate(atoms):
    """ Do the mutation of the atoms input. """

    tc = True
    top_ndx = [atom.index for atom in atoms if atom.symbol=='Pt']# or atom.symbol=='H' or atom.symbol=='C']
    slab_ndx = [atom.index for atom in atoms if atom.symbol!='Pt']# or atom.symbol=='H' or atom.symbol=='C']
    top = atoms[top_ndx]
    slab = atoms[slab_ndx]

    num = top.numbers
    unique_types = list(set(num))
    nu = dict()
    for u in unique_types:
        nu[u] = sum(num == u)

    unique_atom_types = get_all_atom_types(slab, num)
    blmin = closest_distances_generator(atom_numbers=unique_atom_types,
                                    ratio_of_covalent_radii=0.7)
    n_tries = 1000
    counter = 0
    changed = False

    while tc and counter < n_tries:
        counter += 1
        cand = top.copy()
        pos = cand.get_positions()
        cm = np.average(top.get_positions(), axis=0)

        # first select a randomly oriented cutting plane
        theta = pi * np.random.random()
        phi = 2. * pi * np.random.random()
        n = (cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta))
        n = np.array(n)
        # n[2]=0

        # Calculate all atoms signed distance to the cutting plane
        D = []
        for (i, p) in enumerate(pos):
            d = np.dot(p - cm, n)
            D.append((i, d))

        # Sort the atoms by their signed distance
        D.sort(key=lambda x: x[1])
        nu_taken = dict()

        # Select half of the atoms needed for a full cluster
        p_use = []
        n_use = []
        for (i, d) in D:
            if num[i] not in nu_taken.keys():
                nu_taken[num[i]] = 0
            if nu_taken[num[i]] < nu[num[i]] / 2.:
                p_use.append(pos[i])
                n_use.append(num[i])
                nu_taken[num[i]] += 1

        reflect=False
        # calculate the mirrored position and add these.
        pn = []
        for p in p_use:
            pt = p - 2. * np.dot(p - cm, n) * n
            if reflect:
                pt = -pt + 2 * cm + 2 * n * np.dot(pt - cm, n)
            pn.append(pt)

        n_use.extend(n_use)
        p_use.extend(pn)

        # In the case of an uneven number of
        # atoms we need to add one extra
        for n in nu.keys():
            if nu[n] % 2 == 0:
                continue
            while sum(n_use == n) > nu[n]:
                for i in range(int(len(n_use) / 2), len(n_use)):
                    if n_use[i] == n:
                        del p_use[i]
                        del n_use[i]
                        break
            assert sum(n_use == n) == nu[n]

        # Make sure we have the correct number of atoms
        # and rearrange the atoms so they are in the right order
        for i in range(len(n_use)):
            if num[i] == n_use[i]:
                continue
            for j in range(i + 1, len(n_use)):
                if n_use[j] == num[i]:
                    tn = n_use[i]
                    tp = p_use[i]
                    n_use[i] = n_use[j]
                    p_use[i] = p_use[j]
                    p_use[j] = tp
                    n_use[j] = tn

        cand = Atoms(num, p_use, cell=slab.get_cell(), pbc=slab.get_pbc())
        tc = atoms_too_close(cand, blmin)
        if tc:
            continue
        tc = atoms_too_close_two_sets(slab, cand, blmin)

        if not changed and counter > n_tries // 2:
            reflect = not reflect
            changed = True

        tot = slab + cand

    if counter == n_tries:
        return atoms
    return tot

def mirror_mutate_Cu(atoms):
    """ Do the mutation of the atoms input. """

    tc = True
    top_ndx = [atom.index for atom in atoms if atom.symbol=='C' or atom.symbol=='O']# or atom.symbol=='H' or atom.symbol=='C']
    slab_ndx = [atom.index for atom in atoms if atom.symbol=='Cu']# or atom.symbol=='H' or atom.symbol=='C']
    top = atoms[top_ndx]
    slab = atoms[slab_ndx]

    num = top.numbers
    unique_types = list(set(num))
    nu = dict()
    for u in unique_types:
        nu[u] = sum(num == u)

    unique_atom_types = get_all_atom_types(slab, num)
    blmin = closest_distances_generator(atom_numbers=unique_atom_types,
                                    ratio_of_covalent_radii=0.7)
    n_tries = 1000
    counter = 0
    changed = False

    while tc and counter < n_tries:
        counter += 1
        cand = top.copy()
        pos = cand.get_positions()
        cm = np.average(top.get_positions(), axis=0)

        # first select a randomly oriented cutting plane
        theta = pi * np.random.random()
        phi = 2. * pi * np.random.random()
        n = (cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta))
        n = np.array(n)
        # n[2]=0

        # Calculate all atoms signed distance to the cutting plane
        D = []
        for (i, p) in enumerate(pos):
            d = np.dot(p - cm, n)
            D.append((i, d))

        # Sort the atoms by their signed distance
        D.sort(key=lambda x: x[1])
        nu_taken = dict()

        # Select half of the atoms needed for a full cluster
        p_use = []
        n_use = []
        for (i, d) in D:
            if num[i] not in nu_taken.keys():
                nu_taken[num[i]] = 0
            if nu_taken[num[i]] < nu[num[i]] / 2.:
                p_use.append(pos[i])
                n_use.append(num[i])
                nu_taken[num[i]] += 1

        reflect=False
        # calculate the mirrored position and add these.
        pn = []
        for p in p_use:
            pt = p - 2. * np.dot(p - cm, n) * n
            if reflect:
                pt = -pt + 2 * cm + 2 * n * np.dot(pt - cm, n)
            pn.append(pt)

        n_use.extend(n_use)
        p_use.extend(pn)

        # In the case of an uneven number of
        # atoms we need to add one extra
        for n in nu.keys():
            if nu[n] % 2 == 0:
                continue
            while sum(n_use == n) > nu[n]:
                for i in range(int(len(n_use) / 2), len(n_use)):
                    if n_use[i] == n:
                        del p_use[i]
                        del n_use[i]
                        break
            assert sum(n_use == n) == nu[n]

        # Make sure we have the correct number of atoms
        # and rearrange the atoms so they are in the right order
        for i in range(len(n_use)):
            if num[i] == n_use[i]:
                continue
            for j in range(i + 1, len(n_use)):
                if n_use[j] == num[i]:
                    tn = n_use[i]
                    tp = p_use[i]
                    n_use[i] = n_use[j]
                    p_use[i] = p_use[j]
                    p_use[j] = tp
                    n_use[j] = tn

        cand = Atoms(num, p_use, cell=slab.get_cell(), pbc=slab.get_pbc())
        tc = atoms_too_close(cand, blmin)
        if tc:
            continue
        tc = atoms_too_close_two_sets(slab, cand, blmin)

        if not changed and counter > n_tries // 2:
            reflect = not reflect
            changed = True

        tot = slab + cand

    if counter == n_tries:
        return atoms
    return tot
