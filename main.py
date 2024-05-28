from utilities import *
from bio_seq import bio_seq
import os

def main():
    # Step 1: Retrieving the dna sequence
    generate_random = input("Do you have a DNA sequence? If not we will generate one for you. (Y or N): ")
    if generate_random.lower() == "y":
        input_sequence = input("Enter your DNA Sequence: ")
        dna_sequence = bio_seq(input_sequence, "DNA", "MyDNA string\n")
    else:
        dna_sequence = bio_seq(bio_seq.generate_random_sequence(100))
    print(f"\nStep 1 - DNA Sequence: \n{colorize(dna_sequence.sequence)}\n")

    # Step 2: Nucleotides frequency
    nuc_frequency = dna_sequence.count_nuc_frequency()
    print(f"Step 2 - Nucleotides frequency: \n{nuc_frequency}\n")

    # Step 3: Transcription
    rna_sequence = dna_sequence.transcription()
    print(f"Step 3: Transcription: \n{colorize(rna_sequence)}\n")

    # Step 4: Reverse compliments
    reverse_compliments = dna_sequence.reverse_compliment()
    print(f"Step 4: Reverse Compliments: \n5' {colorize(dna_sequence.sequence)} 3'")
    print(f"   {''.join(["|" for i in range(len(dna_sequence.sequence))])}")
    print(f"3' {colorize(reverse_compliments[::-1])} 5' [Compliment]")
    print(f"5' {colorize(reverse_compliments)} 3' [Reverse Compliment]")

    # Step 5: GC Content
    gc_value = dna_sequence.gc_content()
    print(f"\nStep 5: GC Content: \n{gc_value} %")

    # Step 6: GC Sub Content
    gc_k_value = input("\nStep 6: Please enter a GC k-value for your subsets (or default of 20 will be applied): ")
    try:
        gc_sub_value = dna_sequence.gc_content_subsec(gc_k_value)
    except:
        gc_sub_value = dna_sequence.gc_content_subsec()
    print(f"GC Sub content: \n{gc_sub_value}\n")

    # Step 7: DNA Translation
    dna_translation = dna_sequence.translate_seq()
    print(f"Step 7: Translation: \n{dna_translation}\n")

    # Step 8: Codon frequency
    aminoacid = input("Step 8: Please specify a dna aminoacid: ")
    codon_freq = dna_sequence.codon_usage(aminoacid)
    print(f"Codon frequency: \n{codon_freq}\n")

    # Step 9: Reading Frames
    print("\nStep 9: Reading frames for amino acids:")
    for aa_frames in dna_sequence.generate_amino_acid_reading_frames():
        print(aa_frames)

    # Step 10: Proteins
    print("\nStep 10: Proteins from DNA sequence:")
    proteins = dna_sequence.proteins_from_dna_seq()

    if proteins:
        predicted_prot_structs = []
        for protein in proteins:
            print(f"Protein - {protein}")
            try:
                print("Predicting protein structure (using ESMFold API)...")
                predicted_prot_struct = dna_sequence.fold_protein_structure(protein)
                predicted_prot_structs.append(predicted_prot_struct)
                print(f"Predicted protein structure - Successful\n")
            except Exception as er:
                print(f"Predicted protein structure - Failed")
                print(f"Error: {er.args}\n")

        for i in range(len(predicted_prot_structs)):
            write_pdb_file(f"predicted_protein_struct_{i}.pdb", predicted_prot_structs[i])

        print("We have saved all predicted protein structures to the current working directory.\n")
    else:
        print("No proteins found.\n")

    # Step 11: Hamming distance
    hamming_sequence = input(f"Step 11 - Please enter a sequence of length {len(dna_sequence.sequence)}, "
                             f"that will be used to retrieve the hamming distance: ")
    print(f"Hamming distance of sequences: {dna_sequence.hamming_distance(hamming_sequence)}\n")

    # Step 12: Kmers
    kmer = input(f"Step 12 - Please enter a Kmer that will be used to retrieve the Kmer occurrences in the sequence: ")
    print(f"Kmer {kmer} occurrences: {dna_sequence.count_kmer(kmer)}\n")

    # Step 13: Kmers Frequency
    has_random_sequences = input("\nStep 13 - Do you have sequences (fasta file) that you would like to use for "
                                       "detecting the most frequent Kmers? If not we will generate one for you. (Y or N): ")
    if has_random_sequences.lower() == "y":
        fasta_filepath = input("Enter your fasta filepath: ")
        sequences = read_FASTA(fasta_filepath)
    else:
        rnd_seq_num = input("Enter number of random sequences to be generated for frequency detection: ")
        sequences = [bio_seq.generate_random_sequence(len(dna_sequence.sequence)) for i in range(rnd_seq_num)]
    k = int(input(f"Please enter a K value that will be used to get the most frequent Kmers: "))

    freq_kmers = dna_sequence.kmer_frequency(sequences, k)
    print(f"Most frequent Kmers: \n{freq_kmers}")

if __name__ == '__main__':
    main()