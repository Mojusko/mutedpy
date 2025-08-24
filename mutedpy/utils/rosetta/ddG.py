import pyrosetta
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task import operation
import logging
logger = logging.getLogger(__name__)

def rosetta_relax(pose, dump, cartesian=True):
    pyrosetta.init()
    scorefxn = pyrosetta.rosetta.core.scoring.ScoreFunctionFactory.create_score_function("ref2015_cart")

    if cartesian:
        scorefxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreTypeManager.score_type_from_name('cart_bonded'), 0.5)
        # sfxn.set_weight(atom_pair_constraint, 1)#0.5
        scorefxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreTypeManager.score_type_from_name('pro_close'), 0)
        # logger.warning(pyrosetta.rosetta.basic.options.get_boolean_option('ex1'))#set_boolean_option( '-ex1', True )
        # pyrosetta.rosetta.basic.options.set_boolean_option( 'ex2', True )


    # Cloning of the pose including all settings
    working_pose = pose.clone()

    #Define a MoveMap to specify which parts of the structure to minimize
    movemap = pyrosetta.rosetta.core.kinematics.MoveMap()
    movemap.set_bb(True)  # Allow backbone movements
    movemap.set_chi(True)  # Allow sidechain movements

    # Create the MinMover
    min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover()
    min_mover.movemap(movemap)  # Set the MoveMap
    min_mover.score_function(scorefxn)  # Set the score function
    min_mover.min_type('lbfgs_armijo_nonmonotone')  # Set minimization algorithm

    # Apply the MinMover to the pose
    min_mover.apply(working_pose)

    relax = pyrosetta.rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(scorefxn)
    relax.constrain_relax_to_start_coords(True)
    relax.max_iter(200)
    print(relax)

    relax.apply(working_pose)

    #dump the pose
    working_pose.dump_pdb(dump)

def mutate_repack_func(pose, target_positions, residues, repack_radius, sfxn, ddg_bbnbrs=1, verbose=True,
                        cartesian=True, max_iter=None, mers = 4):
    # logger.warning("Interface mode not implemented (should be added!)")
    pyrosetta.init()

    repeats = mers

    if cartesian:
        sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreTypeManager.score_type_from_name('cart_bonded'), 0.5)
        # sfxn.set_weight(atom_pair_constraint, 1)#0.5
        sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreTypeManager.score_type_from_name('pro_close'), 0)
        # logger.warning(pyrosetta.rosetta.basic.options.get_boolean_option('ex1'))#set_boolean_option( '-ex1', True )
        # pyrosetta.rosetta.basic.options.set_boolean_option( 'ex2', True )

    # Cloning of the pose including all settings
    working_pose = pose.clone()

    # Define a MoveMap to specify which parts of the structure to minimize
    # movemap = pyrosetta.rosetta.core.kinematics.MoveMap()
    # movemap.set_bb(True)  # Allow backbone movements
    # movemap.set_chi(True)  # Allow sidechain movements

    # # Create the MinMover
    # min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover()
    # min_mover.movemap(movemap)  # Set the MoveMap
    # min_mover.score_function(sfxn)  # Set the score function
    # min_mover.min_type('lbfgs_armijo_nonmonotone')  # Set minimization algorithm

    # # Apply the MinMover to the pose
    # min_mover.apply(working_pose)

    # relax = pyrosetta.rosetta.protocols.relax.FastRelax()
    # relax.set_scorefxn(sfxn)
    # relax.constrain_relax_to_start_coords(True)
    # relax.max_iter(200)
    # print(relax)

    # relax.apply(working_pose)

    # #dump the pose
    # working_pose.dump_pdb(f"relaxed_mer{repeats}.pdb")

    # Optional: Score the pose after minimization
    score = sfxn(working_pose)
    initial_score = sfxn(pose)
    print("Score before minimization:", initial_score)
    print("Score after minimization:", score)
    # Create all selectors

    i_len = len(pose.sequence()) // repeats
    seq_nbr_or_mutant_selector_LIST_FULL = []
    seq_nbr_or_nbr_or_mutant_selector_LIST_FULL = []

    for target_position, mutant in zip(target_positions, residues):
        seq_nbr_or_mutant_selector_LIST = []
        seq_nbr_or_nbr_or_mutant_selector_LIST = []
        mutant_selector_LIST = []

        for r in range(repeats):
            # Select mutant residue
            mutant_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(
                target_position + i_len * r)

            # Select neighbors with mutant
            nbr_or_mutant_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
            nbr_or_mutant_selector.set_focus(str(target_position))
            nbr_or_mutant_selector.set_distance(repack_radius)
            nbr_or_mutant_selector.set_include_focus_in_subset(True)

            # Select mutant and it's sequence neighbors
            seq_nbr_or_mutant_selector = pyrosetta.rosetta.core.select.residue_selector.PrimarySequenceNeighborhoodSelector(
                ddg_bbnbrs, ddg_bbnbrs, mutant_selector, False)

            # Select mutant, it's seq neighbors and it's surrounding neighbors
            seq_nbr_or_nbr_or_mutant_selector = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector()
            seq_nbr_or_nbr_or_mutant_selector.add_residue_selector(seq_nbr_or_mutant_selector)
            seq_nbr_or_nbr_or_mutant_selector.add_residue_selector(nbr_or_mutant_selector)

            mutant_selector_LIST.append(mutant_selector)
            seq_nbr_or_mutant_selector_LIST.append(seq_nbr_or_mutant_selector)
            seq_nbr_or_nbr_or_mutant_selector_LIST.append(seq_nbr_or_nbr_or_mutant_selector)
            seq_nbr_or_mutant_selector_LIST_FULL.append(seq_nbr_or_mutant_selector)
            seq_nbr_or_nbr_or_mutant_selector_LIST_FULL.append(seq_nbr_or_nbr_or_mutant_selector)

        mutant_selector = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector()
        seq_nbr_or_mutant_selector = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector()
        seq_nbr_or_nbr_or_mutant_selector = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector()

        for i in mutant_selector_LIST:
            mutant_selector.add_residue_selector(i)

        for i in seq_nbr_or_mutant_selector_LIST:
            seq_nbr_or_mutant_selector.add_residue_selector(i)

        for i in seq_nbr_or_nbr_or_mutant_selector_LIST:
            seq_nbr_or_nbr_or_mutant_selector.add_residue_selector(i)

            # Select all except mutant
        all_nand_mutant_selector = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector()
        all_nand_mutant_selector.set_residue_selector(mutant_selector)

        if verbose:
            print(
                f'mutant_selector: {pyrosetta.rosetta.core.select.residue_selector.selection_positions(mutant_selector.apply(working_pose))}')
            print(
                f'all_nand_mutant_selector: {pyrosetta.rosetta.core.select.residue_selector.selection_positions(all_nand_mutant_selector.apply(working_pose))}')
            print(
                f'nbr_or_mutant_selector: {pyrosetta.rosetta.core.select.residue_selector.selection_positions(nbr_or_mutant_selector.apply(working_pose))}')
            print(
                f'seq_nbr_or_mutant_selector: {pyrosetta.rosetta.core.select.residue_selector.selection_positions(seq_nbr_or_mutant_selector.apply(working_pose))}')
            print(
                f'seq_nbr_or_nbr_or_mutant_selector: {pyrosetta.rosetta.core.select.residue_selector.selection_positions(seq_nbr_or_nbr_or_mutant_selector.apply(working_pose))}')

        # Mutate residue and pack rotamers before relax
        # if list(pose.sequence())[target_position-1] != mutant:
        # generate packer task
        tf = TaskFactory()
        tf.push_back(operation.InitializeFromCommandline())
        tf.push_back(operation.IncludeCurrent())

        # Set all residues except mutant to false for design and repacking
        prevent_repacking_rlt = operation.PreventRepackingRLT()
        prevent_subset_repacking = operation.OperateOnResidueSubset(prevent_repacking_rlt, all_nand_mutant_selector,
                                                                    False)
        tf.push_back(prevent_subset_repacking)

        # Assign mutant residue to be designed and repacked
        resfile_comm = pyrosetta.rosetta.protocols.task_operations.ResfileCommandOperation(mutant_selector,
                                                                                           f"PIKAA {mutant}")
        resfile_comm.set_command(f"PIKAA {mutant}")
        tf.push_back(resfile_comm)

        # Apply packing of rotamers of mutant
        packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover()
        packer.score_function(sfxn)
        packer.task_factory(tf)
        if verbose:
            logger.warning(tf.create_task_and_apply_taskoperations(working_pose))
        packer.apply(working_pose)

    seq_nbr_or_mutant_selector = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector()
    seq_nbr_or_nbr_or_mutant_selector = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector()

    for i in seq_nbr_or_mutant_selector_LIST_FULL:
        seq_nbr_or_mutant_selector.add_residue_selector(i)

    for i in seq_nbr_or_nbr_or_mutant_selector_LIST_FULL:
        seq_nbr_or_nbr_or_mutant_selector.add_residue_selector(i)

    if verbose:
        print(
            f'seq_nbr_or_mutant_selector: {pyrosetta.rosetta.core.select.residue_selector.selection_positions(seq_nbr_or_mutant_selector.apply(working_pose))}')
        print(
            f'seq_nbr_or_nbr_or_mutant_selector: {pyrosetta.rosetta.core.select.residue_selector.selection_positions(seq_nbr_or_nbr_or_mutant_selector.apply(working_pose))}')
        for target_position, mutant in zip(target_positions, residues):
            # Get the original residue from the pose (before mutation)
            original_residue = pose.residue(target_position)

            # Get the mutated residue from the working pose (after mutation)
            mutated_residue = working_pose.residue(target_position)

            print(
                f"Original residue at position {target_position}: {original_residue.name()} -> Mutated residue: {mutated_residue.name()} (Desired: {mutant})")

    # allow the movement for bb for the mutant + seq. neighbors, and sc for neigbor in range, seq. neighbor and mutant
    movemap = pyrosetta.rosetta.core.select.movemap.MoveMapFactory()
    movemap.all_jumps(False)
    movemap.add_bb_action(pyrosetta.rosetta.core.select.movemap.mm_enable, seq_nbr_or_mutant_selector)
    movemap.add_chi_action(pyrosetta.rosetta.core.select.movemap.mm_enable, seq_nbr_or_nbr_or_mutant_selector)

    # for checking if all has been selected correctly
    # if verbose:
    mm = movemap.create_movemap_from_pose(working_pose)

    logger.info(mm)

    # Generate a TaskFactory
    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    tf.push_back(operation.IncludeCurrent())
    # tf.push_back(operation.NoRepackDisulfides())

    # prevent all residues except selected from design and repacking
    prevent_repacking_rlt = operation.PreventRepackingRLT()
    prevent_subset_repacking = operation.OperateOnResidueSubset(prevent_repacking_rlt,
                                                                seq_nbr_or_nbr_or_mutant_selector, True)
    tf.push_back(prevent_subset_repacking)

    # allow selected residues only repacking (=switch off design)
    restrict_repacking_rlt = operation.RestrictToRepackingRLT()
    restrict_subset_repacking = operation.OperateOnResidueSubset(restrict_repacking_rlt,
                                                                 seq_nbr_or_nbr_or_mutant_selector, False)
    tf.push_back(restrict_subset_repacking)

    # Perform a FastRelax
    fastrelax = pyrosetta.rosetta.protocols.relax.FastRelax()
    fastrelax.set_scorefxn(sfxn)

    if cartesian:
        fastrelax.cartesian(True)
    if max_iter:
        fastrelax.max_iter(max_iter)

    fastrelax.set_task_factory(tf)
    fastrelax.set_movemap_factory(movemap)
    fastrelax.set_movemap_disables_packing_of_fixed_chi_positions(True)

    if verbose:
        logger.info(tf.create_task_and_apply_taskoperations(working_pose))

    # Score before mutation and repacking
    initial_score = sfxn(pose)
    print("Initial score:", initial_score)

    # Score after mutation and repacking
    intermediate_score = sfxn(working_pose)
    print("Score after mutation and repacking:", intermediate_score)

    fastrelax.apply(working_pose)

    return working_pose