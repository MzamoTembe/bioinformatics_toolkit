import os

letter_colors = {
    "A": "\033[92m",
    "T": "\033[91m",
    "U": "\033[91m",
    "G": "\033[93m",
    "C": "\033[94m",
    "reset": "\033[0;0m"
}

def colorize(string: str) -> str:
    tmpStr = ""

    for s in string:
        if s in letter_colors:
            tmpStr += letter_colors[s] + s
        else:
            tmpStr += letter_colors["reset"] + s

    return tmpStr + "\033[0;0m"

def readFile(file_path: str):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]


def writeFile(file_path: str, seq, mode='w'):
    with open(file_path, mode) as file:
        file.write(seq + '\n')

def read_FASTA(file_path: str):
    with open(file_path, "r") as file:
        return [line.strip() for pos, line in enumerate(file) if pos % 2 != 0]

def write_pdb_file(file_path: str, content: str):
    with open(f"{file_path}.pdb", "w") as pdb_file:
        pdb_file.write(content)
        pdb_file.flush()
        os.fsync(pdb_file.fileno())