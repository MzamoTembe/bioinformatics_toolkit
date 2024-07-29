from Bio.PDB import PDBParser, Polypeptide, Structure
from Bio.PDB.vectors import calc_dihedral, Vector
from scipy.spatial.transform import Rotation
import numpy as np
import argparse
import os

def load_pdb(pdb_file: str) -> Structure:
    """
    Load a PDB file and return the structure.

    Args:
        pdb_file (str): Path to the PDB file.

    Returns:
        Structure: Parsed PDB structure.
    """
    parser = PDBParser(QUIET=True)
    return parser.get_structure('protein', pdb_file)

def get_residue_labels(structure: Structure, chain_id: str) -> np.ndarray:
    """
    Get residue labels for a specific chain in the structure.

    Args:
        structure (Structure): Parsed PDB structure.
        chain_id (str): Chain identifier.

    Returns:
        np.ndarray: Array of residue labels.
    """
    residue_labels = []
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    if Polypeptide.is_aa(residue, standard=True):
                        residue_labels.append(Polypeptide.three_to_index(residue.resname))
    return np.array(residue_labels)

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

def calculate_features(structure: Structure, chain_id: str, neighbor_count: int) -> tuple:
    """
    Calculate structural features for a given chain.

    Args:
        structure (Structure.Structure): Parsed PDB structure.
        chain_id (str): Chain identifier.
        neighbor_count (int): Number of neighbors to consider.

    Returns:
        tuple: Translations, rotations, and torsional angles.
    """
    residues = [residue for chain in structure[0] if chain.id == chain_id
                for residue in chain if Polypeptide.is_aa(residue, standard=True)]
    ca_coords = np.array([residue['CA'].coord for residue in residues])

    basis_vectors = get_basis_vectors(residues, ca_coords)
    distances = np.linalg.norm(ca_coords[:, np.newaxis] - ca_coords, axis=2)
    neighbor_indices = np.argsort(distances, axis=1)[:, 1:neighbor_count + 1]

    translations = calculate_translations(ca_coords, basis_vectors, neighbor_indices)
    rotations = calculate_rotations(basis_vectors, neighbor_indices)
    torsional_angles = calculate_torsional_angles(residues)

    return translations, rotations, torsional_angles

def featurize(pdb_file: str, chain_id: str, neighbor_count: int) -> tuple:
    """
    Featurize a protein chain from a PDB file.

    Args:
        pdb_file (str): Path to the PDB file.
        chain_id (str): Chain identifier.
        neighbor_count (int): Number of neighbors to consider.

    Returns:
        tuple: Residue labels, translations, rotations, and torsional angles.
    """
    structure = load_pdb(pdb_file)
    translations, rotations, torsional_angles = calculate_features(structure, chain_id, neighbor_count)
    residue_labels = get_residue_labels(structure, chain_id)
    return residue_labels, translations, rotations, torsional_angles

def save_features(features: tuple, output_dir: str, filename: str = 'features'):
    """
    Save calculated features to a file.

    Args:
        features (tuple): Residue labels, translations, rotations, and torsional angles.
        output_dir (str): Directory to save the features.
        filename (str): Name of the output file (default: 'features').
    """
    residue_labels, translations, rotations, torsional_angles = features
    output_file = os.path.join(output_dir, filename)
    np.savez(output_file,
             residue_labels=residue_labels,
             translations=translations,
             rotations=rotations,
             torsional_angles=torsional_angles)

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Featurize protein structures from PDB files.")
    parser.add_argument("pdb_file", type=str, help="Path to the PDB file")
    parser.add_argument("chain_id", type=str, help="Chain identifier")
    parser.add_argument("--neighbors", type=int, default=16, help="Number of neighbors to consider")
    parser.add_argument("--output", type=str, default=".", help="Output directory")
    return parser.parse_args()

def main():
    args = parse_arguments()
    features = featurize(args.pdb_file, args.chain_id, args.neighbors)
    save_features(features, args.output, f"{os.path.basename(args.pdb_file)}_{args.chain_id}_features")
    print("Features saved successfully.")

if __name__ == "__main__":
    main()