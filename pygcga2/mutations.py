from ase.data import covalent_radii, chemical_symbols
import numpy as np
from itertools import product
from pygcga2.checkatoms import CheckAtoms
from pygcga2.utilities import NoReasonableStructureFound
from ase.constraints import FixBondLengths, FixedLine
from math import sin, cos, pi, atan2
from pygcga2 import add_molc_on_cluster


BOND_LENGTHS = dict(zip(chemical_symbols, covalent_radii))


def mutation_atoms(atoms=None, dr_percent=0.2,
                   minimum_displacement=0.2,
                   max_trial=500, verbosity=False, elements=None, bond_range=None):
    """
    :param atoms: atoms to be mutated
    :param dr_percent: maximum distance in the perturbation
    :param minimum_displacement:
    :param max_trial: number of trials
    :param verbosity: output verbosity
    :param elements: which elements to be displaced
    :return:
    """
    if atoms is None:
        raise RuntimeError("You are mutating a None type")

    frozed_indexes=[]
    for c in atoms.constraints:
        if isinstance(c, FixBondLengths):
            for indi in c.get_indices():
                frozed_indexes.append(indi)
        elif isinstance(c, FixedLine):
            for indi in c.get_indices():
                frozed_indexes.append(indi)
    symbols = atoms.get_chemical_symbols()
    if elements is None:
        move_elements = list(set(symbols))
    else:
        move_elements = elements[:]
    atoms_checker = CheckAtoms(min_bond=0.5, max_bond=2.0, verbosity=verbosity, bond_range=bond_range)
    for _i_trial in range(max_trial):
        # copy method would copy the constraints too
        a = atoms.copy()
        p0 = a.get_positions()
        dx = np.zeros(shape=p0.shape)
        for j, sj in enumerate(symbols):
            if sj not in move_elements:
                continue
            elif j in frozed_indexes:
                continue
            else:
                rmax = max([BOND_LENGTHS[sj] * dr_percent, minimum_displacement])
                dx[j] = rmax*np.random.uniform(low=-1.0, high=1.0, size=3)
        a.set_positions(p0+dx) # set_positions method would not change the positions of fixed atoms
        if atoms_checker.is_good(a, quickanswer=True):
            return a
        else:
            continue
    raise NoReasonableStructureFound("No reasonable structure found when mutate atoms")


def point_rotate(ep1, ep2, po, theta):
    """
    ep1 first end point of axis
    ep2 second end point of axis

    po point to be rotate
    """
    # translate so axis is at origin
    p=np.asarray(po)-np.asarray(ep1)

    #vector starting with origin
    vector= np.asarray(ep2)-np.asarray(ep1)
    vector= vector / np.sqrt(np.power(vector,2).sum() )

    # matrix common factors
    c = np.cos(theta)
    t = 1.0-np.cos(theta)
    s = np.sin(theta)
    X, Y, Z=vector

    # matrix
    d11 = t*X**2 + c
    d12 = t*X*Y - s*Z
    d13 = t*X*Z + s*Y
    d21 = t*X*Y + s*Z
    d22 = t*Y**2 + c
    d23 = t*Y*Z - s*X
    d31 = t*X*Z - s*Y
    d32 = t*Y*Z + s*X
    d33 = t*Z**2 + c

    x = d11*p[0] + d12*p[1] + d13*p[2]
    y = d21*p[0] + d22*p[1] + d23*p[2]
    z = d31*p[0] + d32*p[1] + d33*p[2]

    return np.asarray([x,y,z]) + np.asarray(ep1)


def rotate_subgroup_atoms(atoms, subgroups, axis_index1, axis_index2, theta):
    """
    :param atoms: atoms object
    :param subgroups: list, the indexes, those atoms will rotate
    :param axis_index1: one point in axis
    :param axis_index2: another one in axis
    :param theta: in degree
    :return:
    """
    assert axis_index1 not in subgroups
    assert axis_index2 not in subgroups
    constraints = atoms.constraints
    positions = atoms.get_positions()
    p1 = positions[axis_index1]
    p2 = positions[axis_index2]

    for index in subgroups:
        positions[index] = point_rotate(p1, p2, positions[index], np.deg2rad(theta))

    atoms.set_constraint()
    atoms.set_positions(positions)
    atoms.set_constraint(constraints)


def cluster_random_perturbation(atoms=None, dr_percent=1.0,
                                minimum_displacement=1.0,
                                max_trial=500, verbosity=False,
                                elements=['Cu', 'Pd'], tags=[], bond_range=None):
    """
    :param atoms: atoms to be mutated
    :param dr_percent: maximum distance in the perturbation
    :param minimum_displacement:
    :param max_trial: number of trials
    :param verbosity: output verbosity
    :param elements: which elements to be perturbed
    :return:
    """
    if atoms is None:
        raise RuntimeError("You are mutating a None type")

    # frozed_indexes=[]
    # for c in atoms.constraints:
    #     if isinstance(c, FixBondLengths):
    #         for indi in c.get_indices():
    #             frozed_indexes.append(indi)
    #     elif isinstance(c, FixedLine):
    #         for indi in c.get_indices():
    #             frozed_indexes.append(indi)
    symbols = atoms.get_chemical_symbols()
    if elements is None:
        move_elements = list(set(symbols))
    else:
        move_elements = elements[:]
    atoms_checker = CheckAtoms(min_bond=0.5, max_bond=2.0, verbosity=verbosity, bond_range=bond_range)
    for _ in range(max_trial):
        # copy method would copy the constraints too
        a = atoms.copy()
        p0 = a.get_positions()
        dx = np.zeros(shape=p0.shape)
        for j, sj in enumerate(symbols):
            if sj not in move_elements and a[j].tag not in tags:
                continue
            # elif j in frozed_indexes:
            #     continue
            else:
                rmax = max([BOND_LENGTHS[sj] * dr_percent, minimum_displacement])
                dx[j] = rmax*np.random.uniform(low=-1.0, high=1.0, size=3)
        a.set_positions(p0+dx) # set_positions method would not change the positions of fixed atoms
        if atoms_checker.is_good(a, quickanswer=True):
            return a
        else:
            continue
    raise NoReasonableStructureFound("No reasonable structure found when mutate atoms")


def cluster_random_displacement(atoms=None,
                                max_trial=500, verbosity=False,
                                elements=['Cu', 'Pd'], tags=[], bond_range=None):
    """
    :param atoms: atoms to be mutated
    :param max_trial: number of trials
    :param verbosity: output verbosity
    :param elements: which elements to be displaced
    :return:
    """
    if atoms is None:
        raise RuntimeError("You are mutating a None type")

    symbols = atoms.get_chemical_symbols()

    if elements is None:
        move_elements = list(set(symbols))
    else:
        move_elements = elements[:]
    atoms_checker = CheckAtoms(min_bond=0.5, max_bond=2.0, verbosity=verbosity, bond_range=bond_range)

    for _ in range(max_trial):
        a = atoms.copy()
        c = atoms.copy()
        cluster, substrate = [], []
        for ai in atoms:
            if ai.symbol in move_elements or ai.tag in tags:
                cluster.append(ai.index)
            else:
                substrate.append(ai.index)
        del a[cluster]
        del c[substrate]
        symbols = c.get_chemical_symbols()
        p0 = c.get_positions()
        Rx, Ry = atoms.cell[0][0], atoms.cell[1][1]
        rx, ry = np.random.uniform(0, Rx), np.random.uniform(0, Ry)
        rz = np.random.uniform(-1.0, 1.0)
        dx = np.full(p0.shape, [rx, ry, rz])

        c.set_positions(p0+dx)

        coord_x, coord_y = [], []
        for atom in c:
            theta_x_i = atom.position[0] / Rx * 2* pi
            theta_y_i = atom.position[1] / Ry * 2* pi
            coord_x.append([cos(theta_x_i), sin(theta_x_i)])
            coord_y.append([cos(theta_y_i), sin(theta_y_i)])
        coord_x = np.array(coord_x)
        coord_y = np.array(coord_y)
        coord_x = np.mean(coord_x, axis=0)
        coord_y = np.mean(coord_y, axis=0)
        theta_x_av = atan2(-np.array(coord_x[1]), -np.array(coord_x[0])) + pi
        theta_y_av = atan2(-np.array(coord_y[1]), -np.array(coord_y[0])) + pi

        xcom = Rx * theta_x_av / (2*pi)
        ycom = Ry * theta_y_av / (2*pi)

        c.positions = c.positions + [Rx/2 - xcom, Ry/2 - ycom, 0]
        c.wrap()
        a.positions = a.positions + [Rx/2 - xcom, Ry/2 - ycom, 0]
        a.wrap()

        ra = np.random.uniform(0, 360)
        c.rotate(ra, (0,0,1), center=(Rx/2, Ry/2, 0))

        a += c
        # a.wrap()
        if atoms_checker.is_good(a, quickanswer=True):
            return a
        else:
            continue
    raise NoReasonableStructureFound("No reasonable structure found when mutate atoms")


def molc_random_displacement(atoms=None,
                            max_trial=500, verbosity=False,
                            tags=[2], bond_range=None):
    """
    :param atoms: atoms to be mutated
    :param max_trial: number of trials
    :param verbosity: output verbosity
    :param elements: which elements to be displaced
    :return:
    """
    if atoms is None:
        raise RuntimeError("You are mutating a None type")
    
    slab = atoms.copy()
    molc = atoms.copy()
    del slab[[atom.index for atom in slab if atom.tag in tags]]
    del molc[[atom.index for atom in molc if atom.tag not in tags]]

    atoms_checker = CheckAtoms(min_bond=0.5, max_bond=2.0, verbosity=verbosity, bond_range=bond_range)

    for _ in range(max_trial):
        cslab = slab.copy()
        cmolc = molc.copy()

        a = add_molc_on_cluster(cslab, cmolc, bond_range=bond_range)

        if atoms_checker.is_good(a, quickanswer=True):
            return a
        else:
            continue
    raise NoReasonableStructureFound("No reasonable structure found when mutate atoms")