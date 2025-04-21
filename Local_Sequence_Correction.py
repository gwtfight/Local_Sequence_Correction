#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Haplotype Matching and Local Sequence Correction
================================================

This script implements two main functionalities:
1. K-mer spectrum-based haplotype matching between GenoPhase and Hifiasm-HiC assemblies
2. Sliding window-based local sequence correction to improve accuracy while maintaining contiguity

Usage:
    python hap_match_and_correct.py --genophase_hapa <path> --genophase_hapb <path> \
                                    --hifiasm_hap1 <path> --hifiasm_hap2 <path> \
                                    [--k_values 21,31,41] [--window_base 10000] \
                                    [--output_dir <path>] [--threads <int>]

Author: Northeast Forestry University 
Date: January , 2025
"""

import os
import sys
import subprocess
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import PairwiseAligner
from scipy.spatial.distance import cosine
import umap
from collections import defaultdict, Counter
import logging
import multiprocessing as mp
from functools import partial
import tempfile
import shutil
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("haplotype_analysis.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class KmerSpectrumAnalyzer:
    """
    Implements k-mer spectrum analysis for haplotype matching between
    GenoPhase and Hifiasm-HiC assemblies.
    """
    
    def __init__(self, haplotypes, k_values, output_dir, threads=1, f_max=1000):
        """
        Initialize the KmerSpectrumAnalyzer.
        
        Parameters:
        -----------
        haplotypes : dict
            Dictionary mapping haplotype names to file paths
        k_values : list
            List of k-mer sizes to analyze
        output_dir : str
            Directory to store results
        threads : int
            Number of threads to use
        f_max : int
            Maximum frequency to consider in k-mer spectrum
        """
        self.haplotypes = haplotypes
        self.k_values = k_values
        self.output_dir = output_dir
        self.threads = threads
        self.f_max = f_max
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create directories for intermediate files
        self.kmer_dir = os.path.join(output_dir, "kmer_counts")
        self.plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.kmer_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Initialize data structures
        self.kmer_spectra = {}  # Will store k-mer spectrum vectors
        self.projections = {}   # Will store UMAP projections
        self.similarity_scores = {}  # Will store similarity scores
        
    def _run_jellyfish(self, fasta_path, k, output_file):
        """
        Run Jellyfish to count k-mers.
        
        Parameters:
        -----------
        fasta_path : str
            Path to FASTA file
        k : int
            k-mer size
        output_file : str
            Output file for jellyfish counts
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        hist_file = f"{output_file}.hist"
        
        # Run jellyfish count
        count_cmd = [
            "jellyfish", "count",
            "-m", str(k),
            "-s", "100M",
            "-t", str(self.threads),
            "-o", output_file,
            fasta_path
        ]
        
        # Run jellyfish histo
        histo_cmd = [
            "jellyfish", "histo",
            "-o", hist_file,
            output_file
        ]
        
        try:
            logger.info(f"Running: {' '.join(count_cmd)}")
            subprocess.run(count_cmd, check=True)
            
            logger.info(f"Running: {' '.join(histo_cmd)}")
            subprocess.run(histo_cmd, check=True)
            
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running Jellyfish: {e}")
            return False
    
    def build_kmer_spectra(self):
        """
        Build k-mer spectrum for each haplotype and each k value.
        """
        logger.info("Building k-mer spectra...")
        
        for hap_name, fasta_path in self.haplotypes.items():
            for k in self.k_values:
                output_file = os.path.join(self.kmer_dir, f"{hap_name}_k{k}")
                hist_file = f"{output_file}.hist"
                
                # Run jellyfish if histogram doesn't exist
                if not os.path.exists(hist_file):
                    success = self._run_jellyfish(fasta_path, k, output_file)
                    if not success:
                        logger.error(f"Failed to build k-mer spectrum for {hap_name} with k={k}")
                        continue
                
                # Load the histogram data
                try:
                    hist_data = pd.read_csv(hist_file, sep=" ", header=None, names=["freq", "count"])
                    
                    # Truncate at f_max
                    hist_data = hist_data[hist_data["freq"] <= self.f_max]
                    
                    # Pad with zeros if needed
                    full_spectrum = np.zeros(self.f_max + 1)
                    full_spectrum[hist_data["freq"].values] = hist_data["count"].values
                    
                    # Store the spectrum
                    if k not in self.kmer_spectra:
                        self.kmer_spectra[k] = {}
                    
                    self.kmer_spectra[k][hap_name] = full_spectrum
                    
                    logger.info(f"Successfully loaded k-mer spectrum for {hap_name} with k={k}")
                except Exception as e:
                    logger.error(f"Error loading k-mer histogram for {hap_name} with k={k}: {e}")
    
    def normalize_and_reduce_dimensions(self):
        """
        Normalize k-mer spectra and reduce dimensions using UMAP.
        """
        logger.info("Normalizing and reducing dimensions...")
        
        for k in self.k_values:
            if k not in self.kmer_spectra:
                logger.warning(f"No k-mer spectra found for k={k}")
                continue
            
            spectra = self.kmer_spectra[k]
            
            # Normalize each spectrum
            normalized_spectra = {}
            for hap_name, spectrum in spectra.items():
                mu = np.mean(spectrum)
                sigma = np.std(spectrum)
                
                # Avoid division by zero
                if sigma == 0:
                    sigma = 1e-10
                
                normalized_spectrum = (spectrum - mu) / sigma
                normalized_spectra[hap_name] = normalized_spectrum
            
            # Prepare data for UMAP
            X = np.vstack([normalized_spectra[hap] for hap in normalized_spectra])
            hap_names = list(normalized_spectra.keys())
            
            # Apply UMAP for dimensionality reduction
            reducer = umap.UMAP(
                n_neighbors=2,
                min_dist=0.1,
                n_components=2,
                random_state=42
            )
            
            embeddings = reducer.fit_transform(X)
            
            # Store the projections
            if k not in self.projections:
                self.projections[k] = {}
            
            for i, hap_name in enumerate(hap_names):
                self.projections[k][hap_name] = embeddings[i]
            
            # Visualize the projections
            plt.figure(figsize=(10, 8))
            for i, hap_name in enumerate(hap_names):
                plt.scatter(embeddings[i, 0], embeddings[i, 1], s=100, label=hap_name)
            
            plt.title(f"UMAP Projection of k-mer Spectra (k={k})")
            plt.xlabel("UMAP Dimension 1")
            plt.ylabel("UMAP Dimension 2")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            plt.savefig(os.path.join(self.plot_dir, f"umap_projection_k{k}.png"), dpi=300)
            plt.close()
    
    def calculate_similarity(self):
        """
        Calculate similarity scores between haplotypes based on their UMAP projections.
        """
        logger.info("Calculating similarity scores...")
        
        # Initialize similarity scores for each k
        for k in self.k_values:
            if k not in self.projections:
                logger.warning(f"No projections found for k={k}")
                continue
            
            projections = self.projections[k]
            hap_names = list(projections.keys())
            
            # Calculate similarity matrix for this k
            sim_matrix = np.zeros((len(hap_names), len(hap_names)))
            
            for i, hap_i in enumerate(hap_names):
                for j, hap_j in enumerate(hap_names):
                    # Calculate cosine similarity
                    p_i = projections[hap_i]
                    p_j = projections[hap_j]
                    
                    # Cosine similarity
                    sim = 1 - cosine(p_i, p_j)  # Convert cosine distance to similarity
                    
                    # Map to [0, 1] range
                    kss = (sim + 1) / 2
                    
                    sim_matrix[i, j] = kss
            
            # Create pandas DataFrame for visualization
            sim_df = pd.DataFrame(sim_matrix, index=hap_names, columns=hap_names)
            
            # Store the similarity scores
            self.similarity_scores[k] = sim_df
            
            # Visualize the similarity matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(sim_df, annot=True, cmap="YlGnBu", vmin=0, vmax=1, 
                        fmt=".2f", linewidths=.5)
            plt.title(f"Haplotype Similarity Matrix (k={k})")
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(self.plot_dir, f"similarity_matrix_k{k}.png"), dpi=300)
            plt.close()
    
    def integrate_similarities(self):
        """
        Integrate similarity scores across different k values.
        """
        logger.info("Integrating similarity scores across k values...")
        
        # Check if we have similarity scores for all k values
        if not all(k in self.similarity_scores for k in self.k_values):
            logger.warning("Missing similarity scores for some k values")
        
        # Get all haplotype names
        hap_names = list(self.similarity_scores[self.k_values[0]].index)
        
        # Initialize final similarity matrix
        final_sim_matrix = np.zeros((len(hap_names), len(hap_names)))
        
        # Average across k values
        for k in self.k_values:
            if k in self.similarity_scores:
                final_sim_matrix += self.similarity_scores[k].values
        
        final_sim_matrix /= len(self.k_values)
        
        # Create pandas DataFrame
        final_sim_df = pd.DataFrame(final_sim_matrix, index=hap_names, columns=hap_names)
        
        # Visualize the final similarity matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(final_sim_df, annot=True, cmap="YlGnBu", vmin=0, vmax=1, 
                    fmt=".2f", linewidths=.5)
        plt.title(f"Integrated Haplotype Similarity Matrix (k={','.join(map(str, self.k_values))})")
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.plot_dir, "integrated_similarity_matrix.png"), dpi=300)
        plt.close()
        
        # Save the matrix to CSV
        final_sim_df.to_csv(os.path.join(self.output_dir, "integrated_similarity_matrix.csv"))
        
        return final_sim_df
    
    def match_haplotypes(self, integrated_sim_df):
        """
        Match GenoPhase haplotypes to Hifiasm-HiC haplotypes based on similarity scores.
        
        Parameters:
        -----------
        integrated_sim_df : pandas.DataFrame
            Integrated similarity matrix
        
        Returns:
        --------
        dict
            Dictionary mapping GenoPhase haplotypes to Hifiasm-HiC haplotypes
        """
        logger.info("Matching haplotypes...")
        
        # Identify GenoPhase and Hifiasm-HiC haplotypes
        genophase_haps = [hap for hap in integrated_sim_df.index if 'hapA' in hap or 'hapB' in hap]
        hifiasm_haps = [hap for hap in integrated_sim_df.index if 'hap1' in hap or 'hap2' in hap]
        
        # Find the best match for each GenoPhase haplotype
        matches = {}
        for gp_hap in genophase_haps:
            # Get similarity scores with Hifiasm-HiC haplotypes
            sim_scores = {hifi_hap: integrated_sim_df.loc[gp_hap, hifi_hap] for hifi_hap in hifiasm_haps}
            
            # Find the best match
            best_match = max(sim_scores, key=sim_scores.get)
            matches[gp_hap] = best_match
            
            logger.info(f"Matched {gp_hap} to {best_match} with similarity score {sim_scores[best_match]:.4f}")
        
        # Save matches to file
        with open(os.path.join(self.output_dir, "haplotype_matches.txt"), 'w') as f:
            for gp_hap, hifi_hap in matches.items():
                f.write(f"{gp_hap}\t{hifi_hap}\t{integrated_sim_df.loc[gp_hap, hifi_hap]:.4f}\n")
        
        return matches
    
    def run(self):
        """
        Run the complete k-mer spectrum analysis workflow.
        """
        # Build k-mer spectra
        self.build_kmer_spectra()
        
        # Normalize and reduce dimensions
        self.normalize_and_reduce_dimensions()
        
        # Calculate similarity scores
        self.calculate_similarity()
        
        # Integrate similarity scores
        integrated_sim_df = self.integrate_similarities()
        
        # Match haplotypes
        matches = self.match_haplotypes(integrated_sim_df)
        
        return matches


class WindowBasedCorrector:
    """
    Implements sliding window-based local sequence correction to improve
    assembly accuracy while maintaining contiguity.
    """
    
    def __init__(self, hifiasm_haps, genophase_haps, matches, output_dir, 
                 window_base=10000, gamma=0.5, threads=1):
        """
        Initialize the WindowBasedCorrector.
        
        Parameters:
        -----------
        hifiasm_haps : dict
            Dictionary mapping Hifiasm haplotype names to file paths
        genophase_haps : dict
            Dictionary mapping GenoPhase haplotype names to file paths
        matches : dict
            Dictionary mapping GenoPhase haplotypes to Hifiasm-HiC haplotypes
        output_dir : str
            Directory to store results
        window_base : int
            Base window size for correction (default: 10kb)
        gamma : float
            Adjustment parameter for window size based on complexity
        threads : int
            Number of threads to use
        """
        self.hifiasm_haps = hifiasm_haps
        self.genophase_haps = genophase_haps
        self.matches = matches
        self.output_dir = output_dir
        self.window_base = window_base
        self.gamma = gamma
        self.threads = threads
        
        # Reverse matches for easy lookup
        self.reverse_matches = {v: k for k, v in matches.items()}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create directories for intermediate files
        self.temp_dir = os.path.join(output_dir, "temp")
        self.alignment_dir = os.path.join(output_dir, "alignments")
        self.correction_dir = os.path.join(output_dir, "corrections")
        self.report_dir = os.path.join(output_dir, "reports")
        
        for directory in [self.temp_dir, self.alignment_dir, self.correction_dir, self.report_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize analysis data
        self.region_complexity = {}
        self.corrected_sequences = {}
        self.iteration_stats = []
    
    def _calculate_region_complexity(self, seq):
        """
        Calculate complexity score for a given sequence region.
        
        Parameters:
        -----------
        seq : str
            DNA sequence
        
        Returns:
        --------
        float
            Complexity score in [0, 1] range
        """
        # Calculate repeat content using a simple proxy
        # In a real implementation, you would use RepeatMasker results
        repeat_content = self._estimate_repeat_content(seq)
        
        # Calculate GC deviation
        gc_content = (seq.count('G') + seq.count('C')) / len(seq)
        avg_gc = 0.42  # This should be calculated from the whole genome
        gc_std = 0.05  # This should be calculated from the whole genome
        gc_dev = min(1.0, abs(gc_content - avg_gc) / gc_std * 2)
        
        # Calculate k-mer diversity
        k_div = self._calculate_kmer_diversity(seq)
        
        # Compute complexity score
        complexity = (repeat_content + gc_dev + k_div) / 3
        
        return complexity
    
    def _estimate_repeat_content(self, seq):
        """
        Estimate repeat content of a sequence.
        This is a simplified proxy for RepeatMasker results.
        
        Parameters:
        -----------
        seq : str
            DNA sequence
        
        Returns:
        --------
        float
            Estimated repeat content in [0, 1] range
        """
        # Count 15-mers and estimate repeats from duplicated k-mers
        k = 15
        kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
        kmer_counts = Counter(kmers)
        
        # Count bases in repeats (k-mers appearing more than once)
        bases_in_repeats = sum(count * k for kmer, count in kmer_counts.items() if count > 1)
        
        # Adjust for overcounting
        estimated_repeat_bases = min(bases_in_repeats, len(seq))
        
        return estimated_repeat_bases / len(seq)
    
    def _calculate_kmer_diversity(self, seq):
        """
        Calculate k-mer diversity for a sequence.
        
        Parameters:
        -----------
        seq : str
            DNA sequence
        
        Returns:
        --------
        float
            K-mer diversity score in [0, 1] range
        """
        k = 10  # Use 10-mers
        
        if len(seq) < k:
            return 0.0
        
        # Generate all k-mers
        kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
        
        # Count unique k-mers
        unique_kmers = set(kmers)
        
        # Calculate diversity
        total_kmers = len(kmers)
        unique_ratio = len(unique_kmers) / total_kmers
        
        # Normalize by log base to account for sequence length
        log_factor = np.log(total_kmers) / np.log(len(seq))
        
        return unique_ratio * log_factor
    
    def _calculate_adaptive_window_size(self, region):
        """
        Calculate adaptive window size based on region complexity.
        
        Parameters:
        -----------
        region : str
            Region identifier
        
        Returns:
        --------
        int
            Adjusted window size
        """
        complexity = self.region_complexity.get(region, 0.5)
        return int(self.window_base * (1 + self.gamma * complexity))
    
    def _extract_sequence_windows(self, hap_file, window_base):
        """
        Extract sequence windows from a FASTA file.
        
        Parameters:
        -----------
        hap_file : str
            Path to FASTA file
        window_base : int
            Base window size
        
        Returns:
        --------
        dict
            Dictionary mapping window IDs to sequences
        """
        windows = {}
        
        for record in SeqIO.parse(hap_file, "fasta"):
            seq_id = record.id
            sequence = str(record.seq)
            
            # Calculate region complexities if not already done
            if seq_id not in self.region_complexity:
                self.region_complexity[seq_id] = self._calculate_region_complexity(sequence)
            
            # Adjust window size based on complexity
            window_size = self._calculate_adaptive_window_size(seq_id)
            step_size = window_size // 2  # 50% overlap
            
            # Extract windows
            for i in range(0, len(sequence) - window_size + 1, step_size):
                window_id = f"{seq_id}:{i}-{i+window_size}"
                windows[window_id] = sequence[i:i+window_size]
        
        return windows
    
    def _calculate_error_probability(self, window_id, hifi_seq, geno_seq):
        """
        Calculate error probability for a window.
        
        Parameters:
        -----------
        window_id : str
            Window identifier
        hifi_seq : str
            Hifiasm sequence
        geno_seq : str
            GenoPhase sequence
        
        Returns:
        --------
        float
            Error probability in [0, 1] range
        """
        # Calculate sequence similarity
        matches = sum(1 for a, b in zip(hifi_seq, geno_seq) if a == b)
        seq_sim = matches / max(len(hifi_seq), len(geno_seq))
        
        # Detect structural variants (simplified)
        sv_score = self._detect_structural_variants(hifi_seq, geno_seq)
        
        # Calculate assembly quality (simplified)
        quality_score = self._estimate_assembly_quality(window_id, hifi_seq)
        
        # Calculate error probability
        alpha1, alpha2, alpha3 = 0.4, 0.3, 0.3
        p_error = alpha1 * (1 - seq_sim) + alpha2 * sv_score + alpha3 * (1 - quality_score)
        
        return p_error
    
    def _detect_structural_variants(self, seq1, seq2):
        """
        Detect structural variants between two sequences.
        This is a simplified implementation.
        
        Parameters:
        -----------
        seq1 : str
            First sequence
        seq2 : str
            Second sequence
        
        Returns:
        --------
        float
            Structural variant score in [0, 1] range
        """
        # Use a simple alignment-based approach
        aligner = PairwiseAligner()
        aligner.mode = 'global'
        aligner.match_score = 2
        aligner.mismatch_score = -1
        aligner.open_gap_score = -2
        aligner.extend_gap_score = -0.5
        
        # Calculate alignment
        alignment = aligner.align(seq1, seq2)[0]
        
        # Extract gap information
        gaps = re.findall(r'-+', str(alignment))
        
        # Calculate SV score based on gap lengths
        if not gaps:
            return 0.0
        
        # Calculate confidence for each gap based on length
        # Longer gaps have higher confidence of being real SVs
        sv_score = 0.0
        total_length = len(seq1) + len(seq2)
        
        for gap in gaps:
            gap_len = len(gap)
            # Weight by gap length - longer gaps are more significant
            gap_weight = min(1.0, gap_len / 50)  # Cap at 50bp
            sv_score += gap_weight * (gap_len / total_length)
        
        return min(1.0, sv_score * 10)  # Scale and cap at 1.0
    
    def _estimate_assembly_quality(self, window_id, seq):
        """
        Estimate assembly quality based on sequence features.
        This is a simplified implementation.
        
        Parameters:
        -----------
        window_id : str
            Window identifier
        seq : str
            Sequence
        
        Returns:
        --------
        float
            Quality score in [0, 1] range
        """
        # In a real implementation, you would use alignment data and quality scores
        # Here we'll use sequence complexity as a proxy
        
        # Calculate k-mer coverage completeness (simplified)
        k_cov = self._calculate_kmer_diversity(seq)
        
        # Estimate depth based on window position (simplified)
        # In real implementation, you would use alignment data
        depth_score = 0.7  # Assume 70% of max depth
        
        # Quality score calculation
        q_score = 0.4 * depth_score + 0.3 * k_cov + 0.3 * (1 - self._estimate_repeat_content(seq))
        
        return q_score
    
    def _calculate_error_threshold(self, region):
        """
        Calculate dynamic error threshold for a region.
        
        Parameters:
        -----------
        region : str
            Region identifier
        
        Returns:
        --------
        float
            Error threshold in [0, 1] range
        """
        # Extract region from window ID
        if ':' in region:
            region = region.split(':')[0]
        
        # Get complexity data
        complexity = self.region_complexity.get(region, 0.5)
        
        # Use a simple adaptive threshold model
        # Higher complexity regions get more lenient thresholds
        mu_r = 0.2  # Base error rate
        sigma_r = 0.05  # Standard deviation
        C = 2.0  # Confidence parameter
        alpha = 0.5  # Weight for repeat density
        beta = 0.3  # Weight for complexity
        
        # Calculate repeat density (simplified)
        d_repeat = complexity * 0.7  # Use complexity as a proxy
        
        # Calculate complexity factor
        f_complexity = complexity
        
        # Calculate threshold
        threshold = mu_r - C * sigma_r + alpha * d_repeat + beta * f_complexity
        
        # Ensure threshold is in [0, 1] range
        return max(0.1, min(0.8, threshold))
    
    def _calculate_position_reliability(self, pos, geno_qual, geno_depth, hifi_qual, hifi_depth):
        """
        Calculate reliability score for a specific position.
        
        Parameters:
        -----------
        pos : int
            Position
        geno_qual : float
            GenoPhase quality score
        geno_depth : int
            GenoPhase sequencing depth
        hifi_qual : float
            Hifiasm quality score
        hifi_depth : int
            Hifiasm sequencing depth
        
        Returns:
        --------
        float
            Reliability score, higher values favor GenoPhase
        """
        # In a real implementation, these values would come from alignments
        # Here we're using placeholder values
        
        # Calculate reliability ratio
        r_diff = (geno_qual * geno_depth) / (max(1, hifi_qual * hifi_depth))
        
        return r_diff
    
    def _calculate_sv_reliability(self, sv, geno_evidence, hifi_evidence, geno_complexity, hifi_complexity):
        """
        Calculate reliability score for a structural variant.
        
        Parameters:
        -----------
        sv : str
            SV identifier
        geno_evidence : float
            Evidence supporting SV in GenoPhase
        hifi_evidence : float
            Evidence supporting SV in Hifiasm
        geno_complexity : float
            Sequence complexity in GenoPhase
        hifi_complexity : float
            Sequence complexity in Hifiasm
        
        Returns:
        --------
        float
            Reliability score, higher values favor GenoPhase
        """
        # Calculate SV reliability
        if hifi_evidence == 0:
            hifi_evidence = 0.001  # Avoid division by zero
        
        if hifi_complexity == 0:
            hifi_complexity = 0.001  # Avoid division by zero
        
        r_sv = (geno_evidence / hifi_evidence) * (geno_complexity / hifi_complexity)
        
        return r_sv
    
    def _create_smooth_transition(self, original_seq, corrected_seq, left_pos, right_pos):
        """
        Create a smooth transition between original and corrected sequences.
        
        Parameters:
        -----------
        original_seq : str
            Original sequence
        corrected_seq : str
            Corrected sequence
        left_pos : int
            Left boundary position
        right_pos : int
            Right boundary position
        
        Returns:
        --------
        str
            Sequence with smooth transition
        """
        # Get transition regions (50bp on each side)
        transition_size = 50
        
        left_orig = original_seq[max(0, left_pos-transition_size):left_pos]
        right_orig = original_seq[right_pos:min(len(original_seq), right_pos+transition_size)]
        
        # Create alignment for smooth transition
        aligner = PairwiseAligner()
        aligner.mode = 'local'
        aligner.match_score = 2
        aligner.mismatch_score = -1
        aligner.open_gap_score = -5
        aligner.extend_gap_score = -1
        
        # Create left transition
        if left_pos > 0:
            left_corr = corrected_seq[:transition_size]
            left_alignment = aligner.align(left_orig, left_corr)[0]
            # Use the alignment to create a smooth transition
            # (In a real implementation, this would be more sophisticated)
            left_transition = left_orig
        else:
            left_transition = ""
        
        # Create right transition
        if right_pos < len(original_seq):
            right_corr = corrected_seq[-transition_size:]
            right_alignment = aligner.align(right_orig, right_corr)[0]
            # Use the alignment to create a smooth transition
            right_transition = right_orig
        else:
            right_transition = ""
        
        # Combine sequences
        final_seq = (original_seq[:max(0, left_pos-len(left_transition))] + 
                    left_transition + 
                    corrected_seq + 
                    right_transition + 
                    original_seq[min(len(original_seq), right_pos+len(right_transition)):])
        
        return final_seq
    
    def _calculate_assembly_quality_gain(self, old_n50, new_n50, old_acc, new_acc):
        """
        Calculate assembly quality gain after correction.
        
        Parameters:
        -----------
        old_n50 : float
            N50 before correction
        new_n50 : float
            N50 after correction
        old_acc : float
            Accuracy before correction
        new_acc : float
            Accuracy after correction
        
        Returns:
        --------
        float
            Assembly quality gain
        """
        # Check for division by zero
        if old_acc >= 1.0:
            acc_gain = 0
        else:
            acc_gain = (new_acc - old_acc) / (1 - old_acc)
        
        # Calculate gain
        gain = (new_n50 / old_n50) * acc_gain
        
        return gain
    
    def _run_sequence_correction(self, hifi_hap, geno_hap):
        """
        Run sequence correction for a pair of haplotypes.
        
        Parameters:
        -----------
        hifi_hap : str
            Hifiasm haplotype name
        geno_hap : str
            GenoPhase haplotype name
        
        Returns:
        --------
        dict
            Dictionary with correction statistics
        """
        logger.info(f"Running sequence correction for {hifi_hap} using {geno_hap}")
        
        # Get file paths
        hifi_file = self.hifiasm_haps[hifi_hap]
        geno_file = self.genophase_haps[geno_hap]
        
        # Initialize corrected sequences
        corrected_records = list(SeqIO.parse(hifi_file, "fasta"))
        
        # Extract windows from both assemblies
        hifi_windows = self._extract_sequence_windows(hifi_file, self.window_base)
        geno_windows = self._extract_sequence_windows(geno_file, self.window_base)
        
        # Initialize statistics
        stats = {
            "iteration": 0,
            "windows_analyzed": len(hifi_windows),
            "windows_corrected": 0,
            "snvs_corrected": 0,
            "indels_corrected": 0,
            "svs_corrected": 0,
            "total_bases_corrected": 0,
            "n50_before": self._calculate_n50([len(str(record.seq)) for record in corrected_records]),
            "n50_after": 0,
            "accuracy_before": 0.95,  # Placeholder
            "accuracy_after": 0.95,   # Placeholder
            "quality_gain": 0.0
        }
        
        # Process each window
        for window_id, hifi_seq in hifi_windows.items():
            # Parse window ID to get contig and position
            if ':' not in window_id or '-' not in window_id:
                continue
                
            contig, pos_range = window_id.split(':')
            start, end = map(int, pos_range.split('-'))
            
            # Find corresponding window in GenoPhase assembly
            # In a real implementation, you would use alignment information
            # Here we're using a simple name-based match
            geno_window_id = None
            for gw_id in geno_windows.keys():
                if contig in gw_id and pos_range in gw_id:
                    geno_window_id = gw_id
                    break
            
            if not geno_window_id:
                continue
                
            geno_seq = geno_windows[geno_window_id]
            
            # Calculate error probability
            p_error = self._calculate_error_probability(window_id, hifi_seq, geno_seq)
            
            # Calculate error threshold
            t_error = self._calculate_error_threshold(window_id)
            
            # Decide whether to correct this window
            if p_error > t_error:
                stats["windows_corrected"] += 1
                
                # Find the contig in corrected records
                for i, record in enumerate(corrected_records):
                    if record.id == contig:
                        # Get the sequence
                        seq = str(record.seq)
                        
                        # Get the region to correct
                        region_to_correct = seq[start:end]
                        
                        # Identify correctable issues
                        # In a real implementation, you would use more sophisticated methods
                        
                        # Correct SNVs
                        snvs = sum(1 for a, b in zip(region_to_correct, geno_seq) if a != b and len(a) == len(b) == 1)
                        stats["snvs_corrected"] += snvs
                        
                        # Correct small indels
                        indels = abs(len(region_to_correct) - len(geno_seq)) - self._detect_structural_variants(region_to_correct, geno_seq) * 100
                        indels = max(0, int(indels))
                        stats["indels_corrected"] += indels
                        
                        # Correct SVs
                        sv_score = self._detect_structural_variants(region_to_correct, geno_seq)
                        if sv_score > 0.2:
                            stats["svs_corrected"] += 1
                        
                        # Create corrected sequence with smooth transitions
                        corrected_seq = self._create_smooth_transition(seq, geno_seq, start, end)
                        
                        # Count corrected bases
                        bases_corrected = sum(1 for a, b in zip(seq, corrected_seq) if a != b)
                        stats["total_bases_corrected"] += bases_corrected
                        
                        # Update the record
                        corrected_records[i].seq = Seq(corrected_seq)
                        break
        
        # Calculate N50 after correction
        stats["n50_after"] = self._calculate_n50([len(str(record.seq)) for record in corrected_records])
        
        # In a real implementation, you would calculate accuracy more precisely
        # Here we're estimating it based on correction percentage
        correction_percentage = stats["total_bases_corrected"] / sum(len(str(record.seq)) for record in corrected_records)
        stats["accuracy_after"] = min(0.999, stats["accuracy_before"] + correction_percentage * 0.05)
        
        # Calculate quality gain
        stats["quality_gain"] = self._calculate_assembly_quality_gain(
            stats["n50_before"], stats["n50_after"], 
            stats["accuracy_before"], stats["accuracy_after"]
        )
        
        # Save corrected sequences
        output_file = os.path.join(self.correction_dir, f"{hifi_hap}_corrected.fasta")
        SeqIO.write(corrected_records, output_file, "fasta")
        
        return stats
    
    def _calculate_n50(self, lengths):
        """
        Calculate N50 for a list of sequence lengths.
        
        Parameters:
        -----------
        lengths : list
            List of sequence lengths
        
        Returns:
        --------
        int
            N50 value
        """
        sorted_lengths = sorted(lengths, reverse=True)
        total_length = sum(sorted_lengths)
        target_length = total_length * 0.5
        
        current_length = 0
        for length in sorted_lengths:
            current_length += length
            if current_length >= target_length:
                return length
        
        return 0
    
    def run_iterative_correction(self, max_iterations=5, min_gain_threshold=0.1):
        """
        Run iterative correction until convergence or max iterations.
        
        Parameters:
        -----------
        max_iterations : int
            Maximum number of iterations
        min_gain_threshold : float
            Minimum quality gain to continue iterations
        
        Returns:
        --------
        dict
            Correction statistics for each haplotype
        """
        logger.info(f"Starting iterative correction with max {max_iterations} iterations")
        
        all_stats = {}
        
        # Process each haplotype pair
        for hifi_hap, geno_hap in self.reverse_matches.items():
            logger.info(f"Processing haplotype pair: {hifi_hap} - {geno_hap}")
            
            # Initialize working copies
            working_dir = os.path.join(self.temp_dir, hifi_hap)
            os.makedirs(working_dir, exist_ok=True)
            
            # Copy original Hifiasm file to working directory
            working_file = os.path.join(working_dir, f"{hifi_hap}_iter0.fasta")
            shutil.copy(self.hifiasm_haps[hifi_hap], working_file)
            
            # Initialize iteration stats
            hap_stats = []
            
            # Run iterations
            for i in range(1, max_iterations + 1):
                logger.info(f"Running iteration {i} for {hifi_hap}")
                
                # Update hifiasm_haps with the latest version
                self.hifiasm_haps[hifi_hap] = working_file
                
                # Run correction
                stats = self._run_sequence_correction(hifi_hap, geno_hap)
                stats["iteration"] = i
                hap_stats.append(stats)
                
                # Update working file for next iteration
                working_file = os.path.join(working_dir, f"{hifi_hap}_iter{i}.fasta")
                shutil.copy(os.path.join(self.correction_dir, f"{hifi_hap}_corrected.fasta"), working_file)
                
                # Check termination conditions
                if i > 1:
                    prev_gain = hap_stats[-2]["quality_gain"]
                    curr_gain = stats["quality_gain"]
                    
                    # Check if gain is diminishing
                    if prev_gain > 0 and (curr_gain - prev_gain) / prev_gain < min_gain_threshold:
                        logger.info(f"Terminating correction for {hifi_hap} due to diminishing returns")
                        break
                
                # Check if too many windows are being corrected
                if stats["windows_corrected"] / stats["windows_analyzed"] > 0.45:
                    logger.warning(f"Terminating correction for {hifi_hap} due to excessive corrections")
                    break
            
            # Store stats for this haplotype
            all_stats[hifi_hap] = hap_stats
            
            # Create a final corrected file
            final_file = os.path.join(self.output_dir, f"{hifi_hap}_final_corrected.fasta")
            shutil.copy(working_file, final_file)
            
            # Generate report
            self._generate_correction_report(hifi_hap, hap_stats)
        
        return all_stats
    
    def _generate_correction_report(self, haplotype, stats):
        """
        Generate a correction report for a haplotype.
        
        Parameters:
        -----------
        haplotype : str
            Haplotype name
        stats : list
            List of statistics for each iteration
        """
        # Create report file
        report_file = os.path.join(self.report_dir, f"{haplotype}_correction_report.txt")
        
        with open(report_file, 'w') as f:
            f.write(f"Correction Report for {haplotype}\n")
            f.write("=" * 50 + "\n\n")
            
            # Write summary
            f.write("Summary:\n")
            f.write(f"Total iterations: {len(stats)}\n")
            f.write(f"Final N50: {stats[-1]['n50_after']}\n")
            f.write(f"Initial N50: {stats[0]['n50_before']}\n")
            f.write(f"Final accuracy estimate: {stats[-1]['accuracy_after']:.4f}\n")
            f.write(f"Total bases corrected: {sum(s['total_bases_corrected'] for s in stats)}\n")
            f.write(f"Total SNVs corrected: {sum(s['snvs_corrected'] for s in stats)}\n")
            f.write(f"Total indels corrected: {sum(s['indels_corrected'] for s in stats)}\n")
            f.write(f"Total SVs corrected: {sum(s['svs_corrected'] for s in stats)}\n\n")
            
            # Write per-iteration details
            f.write("Per-iteration details:\n")
            f.write("-" * 50 + "\n")
            
            for i, s in enumerate(stats):
                f.write(f"Iteration {i+1}:\n")
                f.write(f"  Windows analyzed: {s['windows_analyzed']}\n")
                f.write(f"  Windows corrected: {s['windows_corrected']} ({s['windows_corrected']/s['windows_analyzed']:.2%})\n")
                f.write(f"  Bases corrected: {s['total_bases_corrected']}\n")
                f.write(f"  SNVs corrected: {s['snvs_corrected']}\n")
                f.write(f"  Indels corrected: {s['indels_corrected']}\n")
                f.write(f"  SVs corrected: {s['svs_corrected']}\n")
                f.write(f"  N50 after: {s['n50_after']}\n")
                f.write(f"  Accuracy after: {s['accuracy_after']:.4f}\n")
                f.write(f"  Quality gain: {s['quality_gain']:.4f}\n")
                f.write("-" * 50 + "\n")
        
        # Create visualization of correction progress
        self._visualize_correction_progress(haplotype, stats)
    
    def _visualize_correction_progress(self, haplotype, stats):
        """
        Visualize correction progress for a haplotype.
        
        Parameters:
        -----------
        haplotype : str
            Haplotype name
        stats : list
            List of statistics for each iteration
        """
        # Extract data
        iterations = [s["iteration"] for s in stats]
        windows_corrected_pct = [s["windows_corrected"] / s["windows_analyzed"] * 100 for s in stats]
        n50_values = [s["n50_after"] / stats[0]["n50_before"] for s in stats]
        accuracy_values = [s["accuracy_after"] for s in stats]
        quality_gain = [s["quality_gain"] for s in stats]
        
        # Create plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        
        # Plot windows corrected
        ax1.plot(iterations, windows_corrected_pct, 'o-', color='blue', linewidth=2)
        ax1.set_ylabel('Windows Corrected (%)')
        ax1.set_title(f'Correction Progress for {haplotype}')
        ax1.grid(True, alpha=0.3)
        
        # Plot N50 and accuracy
        ax2.plot(iterations, n50_values, 'o-', color='green', linewidth=2, label='Relative N50')
        ax2.set_ylabel('Relative N50')
        ax2.grid(True, alpha=0.3)
        
        ax2_right = ax2.twinx()
        ax2_right.plot(iterations, accuracy_values, 'o--', color='red', linewidth=2, label='Accuracy')
        ax2_right.set_ylabel('Accuracy')
        
        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_right.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Plot quality gain
        ax3.plot(iterations, quality_gain, 'o-', color='purple', linewidth=2)
        ax3.set_ylabel('Quality Gain')
        ax3.set_xlabel('Iteration')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.report_dir, f"{haplotype}_correction_progress.png"), dpi=300)
        plt.close()


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
    --------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Haplotype Matching and Local Sequence Correction')
    
    # Input files
    parser.add_argument('--genophase_hapa', required=True, help='Path to GenoPhase hapA FASTA file')
    parser.add_argument('--genophase_hapb', required=True, help='Path to GenoPhase hapB FASTA file')
    parser.add_argument('--hifiasm_hap1', required=True, help='Path to Hifiasm hap1 FASTA file')
    parser.add_argument('--hifiasm_hap2', required=True, help='Path to Hifiasm hap2 FASTA file')
    
    # Parameters
    parser.add_argument('--k_values', default='21,31,41', help='Comma-separated list of k-mer sizes')
    parser.add_argument('--window_base', type=int, default=10000, help='Base window size for correction')
    parser.add_argument('--output_dir', default='output', help='Output directory')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads to use')
    
    return parser.parse_args()


def main():
    """Main function to run the haplotype matching and correction workflow."""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare haplotype file paths
    genophase_haps = {
        'hapA': args.genophase_hapa,
        'hapB': args.genophase_hapb
    }
    
    hifiasm_haps = {
        'hap1': args.hifiasm_hap1,
        'hap2': args.hifiasm_hap2
    }
    
    # Parse k values
    k_values = [int(k) for k in args.k_values.split(',')]
    
    # Step 1: K-mer spectrum analysis for haplotype matching
    logger.info("Step 1: Running k-mer spectrum analysis for haplotype matching")
    
    # Combine haplotypes for analyzer
    all_haplotypes = {**genophase_haps, **hifiasm_haps}
    
    analyzer = KmerSpectrumAnalyzer(
        haplotypes=all_haplotypes,
        k_values=k_values,
        output_dir=os.path.join(args.output_dir, "kmer_analysis"),
        threads=args.threads
    )
    
    # Run k-mer spectrum analysis
    matches = analyzer.run()
    
    # Log matches
    logger.info("Haplotype matches:")
    for gp_hap, hifi_hap in matches.items():
        logger.info(f"  {gp_hap} -> {hifi_hap}")
    
    # Step 2: Window-based local sequence correction
    logger.info("Step 2: Running window-based local sequence correction")
    
    corrector = WindowBasedCorrector(
        hifiasm_haps=hifiasm_haps,
        genophase_haps=genophase_haps,
        matches=matches,
        output_dir=os.path.join(args.output_dir, "correction"),
        window_base=args.window_base,
        threads=args.threads
    )
    
    # Run correction
    correction_stats = corrector.run_iterative_correction()
    
    # Log completion
    logger.info("Completed haplotype matching and correction workflow")
    
    # Print summary
    print("\nWorkflow Summary:")
    print("=================")
    print("Haplotype Matches:")
    for gp_hap, hifi_hap in matches.items():
        print(f"  {gp_hap} -> {hifi_hap}")
    
    print("\nCorrection Summary:")
    for hifi_hap, stats_list in correction_stats.items():
        final_stats = stats_list[-1]
        print(f"  {hifi_hap}:")
        print(f"    Iterations: {len(stats_list)}")
        print(f"    Total bases corrected: {sum(s['total_bases_corrected'] for s in stats_list)}")
        print(f"    Final N50: {final_stats['n50_after']}")
        print(f"    Final accuracy estimate: {final_stats['accuracy_after']:.4f}")
    
    print(f"\nDetailed reports and corrected assemblies are available in: {args.output_dir}")


if __name__ == "__main__":
    main()
