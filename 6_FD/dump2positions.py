import numpy as np
import os
import argparse
import logging

argparser = argparse.ArgumentParser()
argparser.add_argument("-f", "--fname", type=str, help="dump.npy")
argparser.add_argument("-k", "--kt", type=str, help="kT")
argparser.add_argument("-s", "--start", type=int, default=0, help="start")
argparser.add_argument("-e", "--end", type=int, default=1000, help="end")
args = argparser.parse_args()

kt = args.kt

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f"log_{kt}_s{args.start}e{args.end}.txt", level=logging.INFO, format="%(message)s")
logging.info(f"kt: {kt}")


def is_bound(arr_DNA: np.ndarray, tf: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    # tf is a single row
    a = np.subtract(arr_DNA, tf)  # distance in each (x,y,z) direction
    distance = np.linalg.norm(a, axis=1)  # euclidian distance

    if np.any(np.less(distance, threshold)):  # if less than threshold distance
        return True
    else:
        return False


def where_bound(
    arr_DNA: np.ndarray, tf: np.ndarray, threshold: float = 0.8
) -> np.ndarray:
    """
    Return the index of the DNA atom where the given tf is bound.
    if tf is not bound to DNA at all, returns np.nan

    """
    n_dna = len(arr_DNA)
    index = np.arange(n_dna).astype(np.uint32)
    # tf is a single row
    a = np.subtract(arr_DNA, tf)  # distance in each (x,y,z) direction
    distance = np.linalg.norm(a, axis=1)  # euclidian distance

    if np.any(np.less(distance, threshold)):
        return index[np.less(distance, threshold)][0]  # only single one
    else:
        return np.nan


def find_positions(arr_DNA: np.ndarray, arr_tf: np.ndarray) -> np.ndarray:
    """
    returns all tfs' position on DNA at a single timestep

    Returns
    -------
    nd.array
        1d array of length = len(arr_tf)
        positions of each tf on DNA polymer
    """

    positions = np.zeros(len(arr_tf))
    for i, tf in enumerate(arr_tf):
        where = where_bound(arr_DNA, tf, threshold=0.9)
        positions[i] = where

    return positions


def find_positions_multistep(
    arr_DNA_ms: np.ndarray, arr_tf_ms: np.ndarray
) -> np.ndarray:
    """
    returns all tfs' position on DNA at all timesteps

    Returns
    -------
    nd.array
        2d array of nrow = n_tf and ncol = n_timestep
        positions of each tf on DNA polymer at all timestep
    """

    n_timestep, n_tf, n_dim = arr_tf_ms.shape
    all_positions = np.zeros(
        [n_tf, n_timestep]
    )  # each col will be a timestep for rows (tfs)

    for i in range(n_timestep):
        all_positions[:, i] = find_positions(arr_DNA_ms[i], arr_tf_ms[i])

        if (i + 1) % 100 == 0:
            log = f"step: {i+1} out of {n_timestep}"
            logger.info(log)

    return all_positions


def save_positions(
    fname: str = "dump.npy", start: int = 4_001, end: int = 20_001, suffix: str = ""
) -> None:
    logger.info("loading data")
    arr = np.load(fname)[start:end]  # timestep between start and end
    logger.info("data loaded")
    n_time, n_atoms, n_dim = arr.shape

    atom_types = arr[0, :, 0]
    atom_types
    atom_id = np.arange(n_atoms) + 1
    n_DNA = sum((atom_types == 2) | (atom_types == 1))  # number of DNA beads
    # n_tf = sum(atom_types == 5)

    logger.info("conditions")
    # conditions
    condition_L = (atom_types == 3) & (atom_id % 3 == 1)
    # condition_H = (atom_types == 5) & (atom_id % 3 == 2)  # second statement in unnecessary
    condition_R = (atom_types == 3) & (atom_id % 3 == 0)

    logger.info("array groups")
    # array gruops
    polymer_DNA = arr[:, :n_DNA, 1:]
    left_legs = arr[:, condition_L, 1:]
    right_legs = arr[:, condition_R, 1:]

    logger.info("finding positions")
    # the positions of each tf for each time step per leg
    logger.info("L")
    positions_L = find_positions_multistep(polymer_DNA, left_legs)
    logging.info("R")
    positions_R = find_positions_multistep(polymer_DNA, right_legs)

    logger.info("saving positions")
    np.savez(f"positions_{suffix}.npz", L=positions_L, R=positions_R)

    return


def main(kt: str):
    UMS = ["10", "20", "40", "60"]

    for um in UMS:
        folder = f"~/5x10t/{kt}/{um}"
        folder = os.path.expanduser(folder)
        os.chdir(folder)
        logging.info(f"working on {folder}")
        save_positions(
            fname=args.fname, start=args.start, end=args.end, suffix=f"s{args.start}e{args.end}"
        )

    return


if __name__ == "__main__":
    main(kt=kt)
