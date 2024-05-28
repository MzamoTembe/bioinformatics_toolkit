import collections
import random
from bio_structs import *
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class bio_seq():
    def __init__(self, sequence="ATGC", seq_type="DNA", label="No label"):
        self.sequence = sequence.upper()
        self.seq_type = seq_type
        self.label = label
        self.is_valid = self.__validate()

    def __validate(self) -> bool:
        """Validates that a Nucleotide sequence contains only valid nucleotides (A, C, G, T). Converts the sequence to uppercase and checks each nucleotide."""
        tmp_seq = self.sequence
        for nuc in tmp_seq:
            if nuc not in NUCLEOTIDE_BASE[self.seq_type]:
                return False
        return True

    def sequence_info(self):
        """Prints the sequence type, length, and label."""
        print(f"Sequence: {self.sequence}")
        print(f"Sequence type: {self.seq_type}")
        print(f"Sequence length: {len(self.sequence)}")
        print(f"Sequence label: {self.label}")

    def generate_random_sequence(length: int, nuc_type="DNA") -> str:
        """Generates a random DNA sequence of the specified length."""
        return "".join(random.choices(NUCLEOTIDE_BASE[nuc_type], k=length))

    def count_nuc_frequency(self) -> dict[str, int]:
        """Counts the frequency of each nucleotide (A, T, G, C) in a DNA sequence. Returns a dictionary with the nucleotide counts."""
        nuc_freq = {"A": 0, "T": 0, "G": 0, "C": 0}
        for nuc in self.sequence:
            nuc_freq[nuc] += 1
        return nuc_freq

    def transcription(self) -> str:
        """Performs transcriptional RNA editing on a DNA sequence, replacing all Ts with Us."""
        if self.seq_type == "DNA":
            return self.sequence.replace("T", "U")
        return "Not a DNA string"

    def reverse_compliment(self, sequence=None) -> str:
        """Takes the reverse complement of a DNA sequence. The reverse complement is the reverse order of nucleotides after replacing each nucleotide with its complement. (A -> T) and (C -> G)"""
        if sequence == None:
            sequence = self.sequence
        return ''.join([reverse_compliments[self.seq_type][nuc] for nuc in sequence])[::-1]

    def gc_content(self, sequence=None) -> int:
        """Calculates the GC content (percentage of G and C nucleotides) of a DNA/RNA sequence."""
        if sequence == None:
            sequence = self.sequence
        lower_sequence = sequence.lower()
        return round(((lower_sequence.count("c") + lower_sequence.count("g")) / len(lower_sequence)) * 100)

    def gc_content_subsec(self, k=10):
        """Calculates the GC content using a sliding window approach across a sequence. Returns a list of GC contents for windows of size k."""
        gc_subsec = []

        for i in range(0, len(self.sequence) - k + 1, k):
            subseq = self.gc_content(self.sequence[i: i + k])
            gc_subsec.append(subseq)

        return gc_subsec

    def translate_seq(self, sequence=None, start_pos=0):
        """Translates a DNA sequence into amino acids by extracting codons and looking them up in a codon table. Returns a list of amino acids."""
        if sequence == None:
            sequence = self.sequence
        if self.seq_type == "DNA":
            return "".join([DNA_Codons[sequence[pos:pos + 3]] for pos in range(start_pos, len(sequence) - 2, 3)])
        else:
            return "".join([RNA_Codons[sequence[pos:pos + 3]] for pos in range(start_pos, len(sequence) - 2, 3)])

    def codon_usage(self, aminoacid: str) -> dict:
        """Calculates the usage frequency of each codon encoding a given amino acid in a sequence. Returns a dictionary of codon:frequency."""
        tmpList = []

        for i in range(0, len(self.sequence) - 2, 3):
            if self.seq_type == "DNA":
                if DNA_Codons[self.sequence[i:i + 3]] == aminoacid.upper():
                    tmpList.append(self.sequence[i:i + 3])
            else:
                if RNA_Codons[self.sequence[i:i + 3]] == aminoacid.upper():
                    tmpList.append(self.sequence[i:i + 3])

        codon_freq = dict(collections.Counter(tmpList))
        total_wight = sum(codon_freq.values())
        for seq in codon_freq:
            codon_freq[seq] = round(codon_freq[seq] / total_wight, 2)

        return codon_freq

    def generate_amino_acid_reading_frames(self, sequence=None):
        """Generates the six reading frames of amino acids from a given DNA sequence by translating it in each of the three forward frames and three reverse complement frames."""
        if sequence == None:
            sequence = self.sequence

        frames = []

        for i in range(4):
            frames.append(self.translate_seq(sequence, i))
            frames.append(self.translate_seq(self.reverse_compliment(sequence), i))

        return frames

    def proteins_from_amino_acid_seq(self, amino_acid_seq: str) -> list[str]:
        """Converts a sequence of amino acids into a list of proteins. A protein is defined as a string of at least 3 amino acids."""
        current_proteins = []
        proteins = []

        for a in amino_acid_seq:
            if a == "_":
                for p in current_proteins:
                    proteins.append(p)
                current_proteins = []
            else:
                if a == "M":
                    current_proteins.append("")
                for i in range(len(current_proteins)):
                    current_proteins[i] += a

        return proteins

    def proteins_from_dna_seq(self, startReadPos=0, endReadPos=0) -> list[str]:
        """Generates a list of all proteins from a given sequence of DNA. The start and end positions of the reading frame can be specified."""
        res = []

        if endReadPos > startReadPos:
            amino_acid_frames = self.generate_amino_acid_reading_frames(self.sequence[startReadPos:endReadPos])
        else:
            amino_acid_frames = self.generate_amino_acid_reading_frames()

        for aa_seq in amino_acid_frames:
            proteins = self.proteins_from_amino_acid_seq(aa_seq)
            for p in proteins:
                res.append(p)

        return res

    def hamming_distance(self, sequence: str) -> int:
        """Calculates the Hamming distance between two DNA sequences. The Hamming distance is the number of nucleotides that differ between two sequences."""
        if len(self.sequence) != len(sequence):
            raise Exception("The two sequences must be of equal length.")

        ham_distance = 0
        for i in range(len(sequence)):
            if self.sequence[i] is not sequence[i]:
                ham_distance += 1

        return ham_distance

    def count_kmer(self, kmer: str) -> int:
        """Counts the number of times a given kmer appears in a sequence."""
        kmer_count = 0
        kmer.upper()
        for position in range(len(self.sequence) - len(kmer) + 1):
            curr_kmer = self.sequence[position: position + len(kmer)]
            if curr_kmer == kmer:
                kmer_count += 1
        return kmer_count

    def kmer_frequency(self, sequences: list[str], k: int) -> list[str]:
        """Calculates the frequency of each kmer in a list of sequences. Returns a list of the most frequent kmers."""
        kmer_freq = {}

        for seq in sequences:
            for pos in range(len(seq) - k + 1):
                kmer = seq[pos:pos + k]
                if kmer in kmer_freq:
                    kmer_freq[kmer] += 1
                else:
                    kmer_freq[kmer] = 1

        max_kmer_freq = max(kmer_freq.values())
        freq_kmers = []

        for kmer in kmer_freq:
            if kmer_freq[kmer] == max_kmer_freq:
                freq_kmers.append(kmer)

        return freq_kmers

    def fold_protein_structure(self, sequence: str) -> str:
        """Gets the protein structure of a given sequence using the ESMAtlas API and returns PDB text"""
        url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        try:
            response = requests.post(url, headers=headers, data=sequence, verify=False)
            response.raise_for_status()
            return response.text

        except Exception as er:
            raise er
