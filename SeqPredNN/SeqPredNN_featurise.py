import os
import csv
import gzip
import argparse
import numpy as np
from Bio.PDB import PDBParser, Polypeptide, Structure
from Bio.PDB.vectors import calc_dihedral, Vector
from scipy.spatial.transform import Rotation

# Constants
DEFAULT_NEIGHBOR_COUNT = 16
DEFAULT_OUTPUT_DIR = "."
CHAIN_LIST_TXT_FILENAME = 'chain_list.txt'

# File handling functions
def parse_arguments():
    parser = argparse.ArgumentParser(description="Featurize protein structures from PDB files.")
    parser.add_argument("pdb_directory", type=str, help="Path to the directory containing PDB files")
    parser.add_argument("--neighbors", type=int, default=DEFAULT_NEIGHBOR_COUNT, help="Number of neighbors to consider")
    parser.add_argument("--chainlist", type=str, default="default",
                        help="Chain list mode: 'default', 'everything', or path to an existing CSV file")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    return parser.parse_args()

def load_pdb(pdb_file: str) -> Structure.Structure:
    parser = PDBParser(QUIET=True)
    if pdb_file.endswith('.gz'):
        with gzip.open(pdb_file, 'rt') as f:
            return parser.get_structure('protein', f)
    else:
        return parser.get_structure('protein', pdb_file)

def get_chain_list(pdb_directory: str, chainlist_option: str) -> list:
    if chainlist_option in ["default", "everything"]:
        chain_list = []
        for filename in os.listdir(pdb_directory):
            if filename.endswith((".pdb", ".pdb.gz")):
                pdb_file = os.path.join(pdb_directory, filename)
                structure = load_pdb(pdb_file)
                protein = os.path.splitext(filename)[0]
                for model in structure:
                    if chainlist_option == "default":
                        chain = next(iter(model))
                        chain_list.append((protein, filename, model.id, chain.id))
                        break
                    elif chainlist_option == "everything":
                        chain_list.extend((protein, filename, model.id, chain.id) for chain in model)
        return chain_list
    with open(chainlist_option, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # Skip header
        return [(row[0], row[1], int(row[2]), row[3]) for row in csv_reader]

def save_chain_list(chain_list: list, output_dir: str):
    output_file = os.path.join(output_dir, CHAIN_LIST_TXT_FILENAME)
    with open(output_file, 'w') as f:
        for protein in chain_list:
            f.write(f"{protein}\n")
    print(f"Chain list saved to {output_file}")

def save_features(features: tuple, output_dir: str, filename: str):
    residue_labels, translations, rotations, torsional_angles = features
    output_file = os.path.join(output_dir, filename)
    np.savez(output_file,
             residue_labels=residue_labels,
             translations=translations,
             rotations=rotations,
             torsional_angles=torsional_angles)

# Utility functions
def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length."""
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)

def dot_product(array_a: np.ndarray, array_b: np.ndarray, keepdims: bool = True) -> np.ndarray:
    """Compute dot product of two arrays."""
    return np.sum(array_a * array_b, axis=-1, keepdims=keepdims)

def project_vectors(array_a: np.ndarray, array_b: np.ndarray, keepdims: bool = True) -> np.ndarray:
    """Project array_a onto array_b."""
    a_dot_b = dot_product(array_a, array_b, keepdims=keepdims)
    b_dot_b = dot_product(array_b, array_b, keepdims=keepdims)
    return a_dot_b / b_dot_b

# Feature calculation functions
def get_residue_labels(structure: Structure.Structure, chain_id: str) -> np.ndarray:
    """
    Get residue labels for a specific chain in the structure.

    Args:
        structure (Structure): Parsed PDB structure.
        chain_id (str): Chain identifier.

    Returns:
        np.ndarray: Array of residue labels.
    """
    residue_labels = []
    for chain in structure:
        if chain.id == chain_id:
            for residue in chain:
                if Polypeptide.is_aa(residue, standard=True):
                    residue_labels.append(Polypeptide.three_to_index(residue.resname))
    return np.array(residue_labels)

def get_basis_vectors(residues: list, ca_coords: np.ndarray) -> np.ndarray:
    """
    Calculate basis vectors for each residue.

    Args:
        residues (list): List of residues.
        ca_coords (np.ndarray): Alpha carbon coordinates.

    Returns:
        np.ndarray: Basis vectors for each residue.
    """
    n_coords = np.array([residue['N'].coord for residue in residues], dtype=float).squeeze()
    c_coords = np.array([residue['C'].coord for residue in residues], dtype=float).squeeze()

    u = normalize_vectors(c_coords - n_coords)
    n_to_ca = normalize_vectors(ca_coords - n_coords)
    v = normalize_vectors(n_to_ca - project_vectors(n_to_ca, u))
    w = np.cross(u, v)
    return np.stack([u, v, w], axis=2)

def calculate_translations(ca_coords: np.ndarray, basis_vectors: np.ndarray,
                           neighbor_indices: np.ndarray) -> np.ndarray:
    """
    Calculate translations in local coordinate systems.

    Args:
        ca_coords (np.ndarray): Alpha carbon coordinates.
        basis_vectors (np.ndarray): Basis vectors for each residue.
        neighbor_indices (np.ndarray): Indices of neighboring residues.

    Returns:
        np.ndarray: Translations in local coordinate systems.
    """
    translations = []
    for i, neighbors in enumerate(neighbor_indices):
        t_vectors = ca_coords[neighbors] - ca_coords[i]

        # Project onto each basis vector
        x_proj = np.dot(t_vectors, basis_vectors[i][:, 0]) / np.dot(basis_vectors[i][:, 0], basis_vectors[i][:, 0])
        y_proj = np.dot(t_vectors, basis_vectors[i][:, 1]) / np.dot(basis_vectors[i][:, 1], basis_vectors[i][:, 1])
        z_proj = np.dot(t_vectors, basis_vectors[i][:, 2]) / np.dot(basis_vectors[i][:, 2], basis_vectors[i][:, 2])

        t_local = np.column_stack((x_proj, y_proj, z_proj))
        translations.append(t_local)
    return np.array(translations)

def calculate_rotations(basis_vectors: np.ndarray, neighbor_indices: np.ndarray) -> np.ndarray:
    """
    Calculate rotations between residues and their neighbors.

    Args:
        basis_vectors (np.ndarray): Basis vectors for each residue.
        neighbor_indices (np.ndarray): Indices of neighboring residues.

    Returns:
        np.ndarray: Rotations as quaternions.
    """
    rotations = []
    for i, neighbors in enumerate(neighbor_indices):
        r_matrices = np.dot(basis_vectors[neighbors], basis_vectors[i].T)
        r_quaternions = Rotation.from_matrix(r_matrices).as_quat()
        rotations.append(r_quaternions)
    return np.array(rotations)

def calculate_torsional_angles(residues: list) -> np.ndarray:
    """
    Calculate phi and psi torsional angles for residues.

    Args:
        residues (list): List of residues.

    Returns:
        np.ndarray: Array of sin and cos of phi and psi angles.
    """
    torsional_angles = []
    for i, residue in enumerate(residues):
        phi = psi = 0
        if i > 0:
            phi = calc_dihedral(
                Vector(residues[i - 1]['C'].coord),
                Vector(residue['N'].coord),
                Vector(residue['CA'].coord),
                Vector(residue['C'].coord)
            )
        if i < len(residues) - 1:
            psi = calc_dihedral(
                Vector(residue['N'].coord),
                Vector(residue['CA'].coord),
                Vector(residue['C'].coord),
                Vector(residues[i + 1]['N'].coord)
            )
        torsional_angles.append((
            [np.sin(phi), np.sin(psi)],
            [np.cos(phi), np.cos(psi)]
        ))
    return np.array(torsional_angles)

def calculate_features(structure: Structure.Structure, model_id: int, chain_id: str, neighbor_count: int) -> tuple:
    """
    Calculate structural features for a given model and chain.

    Args:
        structure (Structure.Structure): Parsed PDB structure.
        model_id (int): Model identifier.
        chain_id (str): Chain identifier.
        neighbor_count (int): Number of neighbors to consider.

    Returns:
        tuple: Residue labels, translations, rotations, and torsional angles.
    """
    residues = [residue for chain in structure[model_id] if chain.id == chain_id
                for residue in chain if Polypeptide.is_aa(residue, standard=True)]
    ca_coords = np.array([residue['CA'].coord for residue in residues])

    basis_vectors = get_basis_vectors(residues, ca_coords)
    distances = np.linalg.norm(ca_coords[:, np.newaxis] - ca_coords, axis=2)
    neighbor_indices = np.argsort(distances, axis=1)[:, 1:neighbor_count + 1]

    translations = calculate_translations(ca_coords, basis_vectors, neighbor_indices)
    rotations = calculate_rotations(basis_vectors, neighbor_indices)
    torsional_angles = calculate_torsional_angles(residues)
    residue_labels = get_residue_labels(structure[model_id], chain_id)

    return residue_labels, translations, rotations, torsional_angles

# Main processing functions
def featurize_proteins(pdb_directory: str, chain_list: list, neighbor_count: int, output_dir: str):
    successful_proteins = []
    for protein, filename, model_id, chain_id in chain_list:
        pdb_file = os.path.join(pdb_directory, filename)
        try:
            structure = load_pdb(pdb_file)
            features = calculate_features(structure, model_id, chain_id, neighbor_count)
            save_features(features, output_dir, f"{protein}_{model_id}_{chain_id}")
            successful_proteins.append(f"{protein}_{model_id}_{chain_id}")
            print(f"Features saved for {protein} (Model - {model_id}, Chain - {chain_id})")
        except Exception as e:
            print(f"Error processing {pdb_file}: {str(e)}")
    save_chain_list(successful_proteins, output_dir)

def main():
    args = parse_arguments()
    os.makedirs(args.output, exist_ok=True)
    chain_list = get_chain_list(args.pdb_directory, args.chainlist)
    featurize_proteins(args.pdb_directory, chain_list, args.neighbors, args.output)
    print("All features saved successfully.")

if __name__ == "__main__":
    main()