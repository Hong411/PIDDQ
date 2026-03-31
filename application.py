# application.py
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from utils import load_dict, GPUCalculator, save_dict
from utils.config import PATH_CONFIG
from collections import defaultdict
import os
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

class InfraredSpectrumApplication:
    """
    Application analysis for infrared spectrum prediction
    Includes spectrum retrieval and functional group analysis
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize the application analyzer
        
        Args:
            use_gpu: Whether to use GPU for calculations
        """
        self.use_gpu = use_gpu
        if use_gpu:
            self.gpu_calc = GPUCalculator()
    
    def calculate_pcc_matrix(self, query_spectra: np.ndarray, database_spectra: np.ndarray) -> np.ndarray:
        """
        Calculate PCC matrix between query spectra and database spectra
        
        Args:
            query_spectra: Query spectra (n_queries × n_features)
            database_spectra: Database spectra (n_db × n_features)
        
        Returns:
            PCC matrix (n_queries × n_db)
        """
        if self.use_gpu:
            pcc_matrix = self.gpu_calc.pearson_matrix_gpu(query_spectra, database_spectra)
        else:
            n_queries = len(query_spectra)
            n_db = len(database_spectra)
            pcc_matrix = np.zeros((n_queries, n_db))
            
            for i in range(n_queries):
                for j in range(n_db):
                    corr = np.corrcoef(query_spectra[i], database_spectra[j])[0, 1]
                    pcc_matrix[i, j] = corr if not np.isnan(corr) else 0.0
        
        return pcc_matrix
    
    def rank_analysis(self, pcc_matrix: np.ndarray, true_indices: np.ndarray, 
                      rank_top: int = 20) -> Dict:
        """
        Perform rank analysis to evaluate retrieval success rate
        
        Args:
            pcc_matrix: PCC matrix (n_queries × n_db)
            true_indices: True matching indices for each query
            rank_top: Maximum rank to consider
        
        Returns:
            Dictionary with rank statistics
        """
        n_queries = len(pcc_matrix)
        ranks = []
        
        for i in range(n_queries):
            scores = pcc_matrix[i]
            true_idx = true_indices[i]
            
            sorted_indices = np.argsort(-scores)
            rank = np.where(sorted_indices == true_idx)[0][0] + 1
            ranks.append(rank)
        
        success_rates = []
        for rank in range(1, rank_top + 1):
            success_count = sum(1 for r in ranks if r <= rank)
            success_rate = success_count / n_queries
            success_rates.append(success_rate)
        
        return {
            'ranks': np.array(ranks),
            'success_rates': np.array(success_rates),
            'rank_top': rank_top
        }
    
    def single_spectrum_retrieval(self, query_spectrum: np.ndarray, 
                                   database_spectra: np.ndarray,
                                   database_ids: List[str],
                                   database_smiles: List[str],
                                   true_id: str = None) -> Dict:
        """
        Retrieve the most similar spectra for a single query spectrum
        
        Args:
            query_spectrum: Single query spectrum (n_features,)
            database_spectra: Database spectra (n_db × n_features)
            database_ids: IDs for database spectra
            database_smiles: SMILES strings for database spectra
            true_id: True ID of the query (if known)
        
        Returns:
            Dictionary with retrieval results
        """
        # Reshape query to 2D
        query_2d = query_spectrum.reshape(1, -1)
        
        # Calculate PCC with all database spectra
        pcc_scores = self.calculate_pcc_matrix(query_2d, database_spectra).flatten()
        
        # Get top matches
        sorted_indices = np.argsort(-pcc_scores)
        top_indices = sorted_indices[:10]  # Top 10 matches
        top_scores = pcc_scores[top_indices]
        top_ids = [database_ids[i] for i in top_indices]
        top_smiles = [database_smiles[i] for i in top_indices]
        
        # Find rank of true ID if provided
        rank = None
        if true_id is not None:
            try:
                true_idx = database_ids.index(true_id)
                rank = np.where(sorted_indices == true_idx)[0][0] + 1
            except ValueError:
                rank = None
        
        results = {
            'top_matches': [
                {'rank': i+1, 'id': top_ids[i], 'smiles': top_smiles[i], 'pcc': top_scores[i]}
                for i in range(len(top_indices))
            ],
            'all_scores': pcc_scores,
            'top_indices': top_indices.tolist(),
            'rank_of_true': rank
        }
        
        return results
    
    def spectrum_retrieval_analysis(self, 
                                     query_spectra: np.ndarray, 
                                     database_spectra: np.ndarray,
                                     query_ids: List[str],
                                     database_ids: List[str],
                                     smiles_list: List[str],
                                     true_indices: np.ndarray = None,
                                     rank_top: int = 20) -> Dict:
        """
        Complete spectrum retrieval analysis
        
        Args:
            query_spectra: Query spectra (n_queries × n_features)
            database_spectra: Database spectra (n_db × n_features)
            query_ids: IDs for query spectra
            database_ids: IDs for database spectra
            smiles_list: SMILES strings for all spectra
            true_indices: True matching indices (if None, assumes query indices match database indices)
            rank_top: Maximum rank for analysis
        
        Returns:
            Dictionary with retrieval results
        """
        print("="*60)
        print("Spectrum Retrieval Analysis")
        print("="*60)
        print(f"Query spectra: {len(query_spectra)}")
        print(f"Database spectra: {len(database_spectra)}")
        
        print("\nCalculating PCC matrix...")
        pcc_matrix = self.calculate_pcc_matrix(query_spectra, database_spectra)
        
        if true_indices is None:
            true_indices = np.arange(len(query_spectra))
        
        print("\nPerforming rank analysis...")
        rank_results = self.rank_analysis(pcc_matrix, true_indices, rank_top)
        
        top1_indices = np.argmax(pcc_matrix, axis=1)
        
        print("\nPerforming functional group analysis...")
        fg_stats = self.functional_group_analysis(smiles_list, top1_indices, top_k=1)
        
        print("\n" + "="*60)
        print("Retrieval Results")
        print("="*60)
        print(f"\nRank Analysis (Top {rank_top}):")
        for rank in [1, 5, 10, 20]:
            if rank <= rank_top:
                rate = rank_results['success_rates'][rank-1]
                print(f"  Top-{rank} success rate: {rate:.4f} ({rate*100:.1f}%)")
        
        print(f"\nFunctional Group Statistics (Top-1 matches):")
        print(f"{'Functional Group':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-"*61)
        for fg, stats in sorted(fg_stats.items(), key=lambda x: x[1]['f1_score'], reverse=True)[:15]:
            print(f"{fg:<25} {stats['precision']:<12.4f} {stats['recall']:<12.4f} {stats['f1_score']:<12.4f}")
        
        results = {
            'pcc_matrix': pcc_matrix,
            'rank_analysis': rank_results,
            'functional_group_stats': fg_stats,
            'top1_indices': top1_indices.tolist()
        }
        
        return results
    
    def functional_group_analysis(self, smiles_list: List[str], 
                                  matched_indices: List[int],
                                  top_k: int = 1) -> Dict:
        """Analyze functional groups for top-k matched spectra"""
        
        def extract_functional_groups(smiles: str) -> List[str]:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []
            
            fgs = []
            fg_patterns = {
                'Alkene': '[CX3]=[CX3]',
                'Alkyne': '[CX2]#C',
                'Aromatic': '[$([cX3](:*):*),$([cX2+](:*):*)]',
                'Alcohol': '[#6][OX2H]',
                'Ester': '[CX3](=O)[OX2H0][#6]',
                'Aldehyde': '[CX3H1](=O)[#6]',
                'Ketone': '[#6][CX3](=O)[#6]',
                'Carboxylic Acid': '[CX3](=O)[OX1H0-,X2H1]',
                'Ether': '[OX2;!$(OC=O)]([#6])[#6]',
                'Amide': '[CX3](=[OX1])[NX3H2,NX3H1,NX3H0,NX4H]',
                'Amine': '[NX3;H2,H1,H0;!$(NC=O)]',
                'Nitrile': '[NX1]#[CX2]',
                'Nitro': '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]',
                'Imine': '[CX3]=[NX2]'
            }
            
            for fg_name, smarts in fg_patterns.items():
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is not None and mol.HasSubstructMatch(pattern):
                    fgs.append(fg_name)
            
            return fgs
        
        fg_correct = defaultdict(int)
        fg_predicted = defaultdict(int)
        fg_actual = defaultdict(int)
        
        for query_idx, matched_idx in enumerate(matched_indices):
            actual_fgs = extract_functional_groups(smiles_list[query_idx])
            predicted_fgs = extract_functional_groups(smiles_list[matched_idx])
            
            for fg in actual_fgs:
                fg_actual[fg] += 1
            for fg in predicted_fgs:
                fg_predicted[fg] += 1
            for fg in actual_fgs:
                if fg in predicted_fgs:
                    fg_correct[fg] += 1
        
        fg_stats = {}
        all_fgs = set(fg_actual.keys()) | set(fg_predicted.keys())
        
        for fg in all_fgs:
            tp = fg_correct.get(fg, 0)
            fp = fg_predicted.get(fg, 0) - tp
            fn = fg_actual.get(fg, 0) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            fg_stats[fg] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn
            }
        
        return fg_stats
    
    def plot_rank_curve(self, rank_results: Dict, save_name: str = "rank_curve"):
        """Plot rank success rate curve"""
        ranks = np.arange(1, rank_results['rank_top'] + 1)
        success_rates = rank_results['success_rates']
        
        width_inch = 8 / 2.54
        height_inch = 6 / 2.54
        
        fig, ax = plt.subplots(figsize=(width_inch, height_inch))
        
        for i in range(1, 11):
            i_s = i / 10
            ax.axhline(i_s, color='gray', alpha=0.3, linestyle='--', linewidth=0.5)
        
        ax.step(ranks, success_rates, label='Success Rate', 
                color='#33ABC1', linewidth=2, where='mid')
        
        ax.set_xlabel('Rank', fontsize=8)
        ax.set_ylabel('Success Rate', fontsize=8)
        ax.set_xlim(0, rank_results['rank_top'] + 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        
        text = f"Top-1: {success_rates[0]:.3f}\nTop-5: {success_rates[4]:.3f}\nTop-10: {success_rates[9]:.3f}"
        ax.text(0.95, 0.05, text, transform=ax.transAxes, fontsize=6,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        
        results_dir = os.path.join(PATH_CONFIG.RESULTS_DIR, 'application')
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, f"{save_name}.pdf")
        fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
        
        plt.show()
        plt.close(fig)
    
    def plot_functional_group_bar(self, fg_stats: Dict, top_n: int = 10, 
                                  save_name: str = "fg_performance"):
        """Plot functional group performance bar chart"""
        sorted_fgs = sorted(fg_stats.items(), key=lambda x: x[1]['f1_score'], reverse=True)[:top_n]
        
        fg_names = [fg[0] for fg in sorted_fgs]
        f1_scores = [fg[1]['f1_score'] for fg in sorted_fgs]
        precision = [fg[1]['precision'] for fg in sorted_fgs]
        recall = [fg[1]['recall'] for fg in sorted_fgs]
        
        width_inch = 10 / 2.54
        height_inch = 6 / 2.54
        
        fig, ax = plt.subplots(figsize=(width_inch, height_inch))
        
        x = np.arange(len(fg_names))
        width = 0.25
        
        ax.bar(x - width, precision, width, label='Precision', color='#7ABBDB')
        ax.bar(x, recall, width, label='Recall', color='#84BA42')
        ax.bar(x + width, f1_scores, width, label='F1-Score', color='#A51C36')
        
        ax.set_xlabel('Functional Group', fontsize=8)
        ax.set_ylabel('Score', fontsize=8)
        ax.set_title('Functional Group Recognition Performance', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(fg_names, rotation=45, ha='right', fontsize=7)
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        results_dir = os.path.join(PATH_CONFIG.RESULTS_DIR, 'application')
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, f"{save_name}.pdf")
        fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
        
        plt.show()
        plt.close(fig)
        
    def classify_molecules_by_elements(self, id_list: List[str], smiles_list: List[str], 
                                        sim_list: np.ndarray = None) -> Dict:
        """
        Classify molecules based on element composition
        
        Args:
            id_list: List of molecule IDs
            smiles_list: List of SMILES strings
            sim_list: Optional list of spectra for each molecule
        
        Returns:
            Dictionary with classification results
        """
        from rdkit import Chem
        
        classification = {
            "CH": {"ids": [], "sim": [] if sim_list is not None else None},
            "CHO": {"ids": [], "sim": [] if sim_list is not None else None},
            "CHON": {"ids": [], "sim": [] if sim_list is not None else None}
        }
        
        for idx, (id_, smiles) in enumerate(zip(id_list, smiles_list)):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    print(f"Warning: {id_} mol not found")
                    continue
                
                elements = set(atom.GetSymbol() for atom in mol.GetAtoms())
                
                if 'N' in elements:
                    classification["CHON"]["ids"].append(id_)
                    if sim_list is not None:
                        classification["CHON"]["sim"].append(sim_list[idx])
                elif 'O' in elements:
                    classification["CHO"]["ids"].append(id_)
                    if sim_list is not None:
                        classification["CHO"]["sim"].append(sim_list[idx])
                else:
                    classification["CH"]["ids"].append(id_)
                    if sim_list is not None:
                        classification["CH"]["sim"].append(sim_list[idx])
            except Exception as e:
                print(f"Warning: {id_} not classified - {e}")
                continue
        
        # Print statistics
        print("\n" + "="*60)
        print("Molecule Classification by Element Composition")
        print("="*60)
        print(f"CH (C,H only): {len(classification['CH']['ids'])} molecules")
        print(f"CHO (C,H,O): {len(classification['CHO']['ids'])} molecules")
        print(f"CHON (C,H,O,N): {len(classification['CHON']['ids'])} molecules")
        
        return classification
    
    def functional_group_analysis_with_element_filter(
        self, 
        smiles_list: List[str], 
        matched_indices: List[int],
        classification: Dict = None
    ) -> Dict[str, Dict]:
        """
        Analyze functional groups with element-based filtering
        
        Args:
            smiles_list: List of SMILES strings
            matched_indices: Indices of matched spectra for each query
            classification: Optional classification dict for element-based filtering
        
        Returns:
            Dictionary with functional group statistics
        """
        
        def extract_functional_groups(smiles: str, elements: set = None) -> List[str]:
            """Extract functional groups with element-based filtering"""
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []
            
            # Get elements if not provided
            if elements is None:
                elements = set(atom.GetSymbol() for atom in mol.GetAtoms())
            
            fgs = []
            
            # Define functional group SMARTS patterns with element requirements
            fg_patterns = {
                'Alkene': {'smarts': '[CX3]=[CX3]', 'requires': set(), 'excludes': set()},
                'Alkyne': {'smarts': '[CX2]#C', 'requires': set(), 'excludes': set()},
                'Aromatic': {'smarts': '[$([cX3](:*):*),$([cX2+](:*):*)]', 'requires': set(), 'excludes': set()},
                'Alcohol': {'smarts': '[#6][OX2H]', 'requires': {'O'}, 'excludes': set()},
                'Ester': {'smarts': '[CX3](=O)[OX2H0][#6]', 'requires': {'O'}, 'excludes': set()},
                'Aldehyde': {'smarts': '[CX3H1](=O)[#6]', 'requires': {'O'}, 'excludes': set()},
                'Ketone': {'smarts': '[#6][CX3](=O)[#6]', 'requires': {'O'}, 'excludes': set()},
                'Carboxylic Acid': {'smarts': '[CX3](=O)[OX1H0-,X2H1]', 'requires': {'O'}, 'excludes': set()},
                'Ether': {'smarts': '[OX2;!$(OC=O)]([#6])[#6]', 'requires': {'O'}, 'excludes': set()},
                'Amide': {'smarts': '[CX3](=[OX1])[NX3H2,NX3H1,NX3H0,NX4H]', 'requires': {'O', 'N'}, 'excludes': set()},
                'Amine': {'smarts': '[NX3;H2,H1,H0;!$(NC=O)]', 'requires': {'N'}, 'excludes': set()},
                'Nitrile': {'smarts': '[NX1]#[CX2]', 'requires': {'N'}, 'excludes': set()},
                'Nitro': {'smarts': '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]', 'requires': {'N', 'O'}, 'excludes': set()},
                'Imine': {'smarts': '[CX3]=[NX2]', 'requires': {'N'}, 'excludes': set()}
            }
            
            for fg_name, fg_info in fg_patterns.items():
                # Check element requirements
                if fg_info['requires']:
                    if not fg_info['requires'].issubset(elements):
                        continue
                
                pattern = Chem.MolFromSmarts(fg_info['smarts'])
                if pattern is not None and mol.HasSubstructMatch(pattern):
                    fgs.append(fg_name)
            
            return fgs
        
        # Count functional groups with element filtering
        fg_correct = defaultdict(int)
        fg_predicted = defaultdict(int)
        fg_actual = defaultdict(int)
        
        for query_idx, matched_idx in enumerate(matched_indices):
            # Get query molecule elements
            query_mol = Chem.MolFromSmiles(smiles_list[query_idx])
            if query_mol is None:
                continue
            query_elements = set(atom.GetSymbol() for atom in query_mol.GetAtoms())
            
            # Get matched molecule elements
            matched_mol = Chem.MolFromSmiles(smiles_list[matched_idx])
            if matched_mol is None:
                continue
            matched_elements = set(atom.GetSymbol() for atom in matched_mol.GetAtoms())
            
            # Extract functional groups with element filtering
            actual_fgs = extract_functional_groups(smiles_list[query_idx], query_elements)
            predicted_fgs = extract_functional_groups(smiles_list[matched_idx], matched_elements)
            
            # Count
            for fg in actual_fgs:
                fg_actual[fg] += 1
            for fg in predicted_fgs:
                fg_predicted[fg] += 1
            for fg in actual_fgs:
                if fg in predicted_fgs:
                    fg_correct[fg] += 1
        
        # Calculate statistics
        fg_stats = {}
        all_fgs = set(fg_actual.keys()) | set(fg_predicted.keys())
        
        for fg in all_fgs:
            tp = fg_correct.get(fg, 0)
            fp = fg_predicted.get(fg, 0) - tp
            fn = fg_actual.get(fg, 0) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            fg_stats[fg] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn
            }
        
        return fg_stats
    
    def rank_analysis_with_mw_filter(
        self,
        pcc_matrix: np.ndarray,
        true_indices: np.ndarray,
        mw_list: List[float],
        mw_tolerance: float = 5.0,
        rank_top: int = 20
    ) -> Dict:
        """
        Perform rank analysis with molecular weight filtering
        
        Args:
            pcc_matrix: PCC matrix (n_queries × n_db)
            true_indices: True matching indices for each query
            mw_list: Molecular weights for database spectra
            mw_tolerance: Tolerance for molecular weight matching (± Da)
            rank_top: Maximum rank to consider
        
        Returns:
            Dictionary with rank statistics
        """
        n_queries = len(pcc_matrix)
        ranks = []
        
        for i in range(n_queries):
            scores = pcc_matrix[i]
            true_idx = true_indices[i]
            true_mw = mw_list[true_idx]
            
            # Filter by molecular weight
            mw_mask = np.abs(np.array(mw_list) - true_mw) <= mw_tolerance
            if not np.any(mw_mask):
                # No molecules within tolerance, use all
                filtered_scores = scores
            else:
                # Set scores outside tolerance to -inf
                filtered_scores = scores.copy()
                filtered_scores[~mw_mask] = -np.inf
            
            # Get ranking within filtered set
            sorted_indices = np.argsort(-filtered_scores)
            
            # Check if true index is still in filtered set
            if true_idx not in sorted_indices[:len(sorted_indices)]:
                rank = len(sorted_indices) + 1  # Not found
            else:
                rank = np.where(sorted_indices == true_idx)[0][0] + 1
            
            ranks.append(rank)
        
        # Calculate success rate at each rank
        success_rates = []
        for rank in range(1, rank_top + 1):
            success_count = sum(1 for r in ranks if r <= rank)
            success_rate = success_count / n_queries
            success_rates.append(success_rate)
        
        # Print statistics
        print(f"\nMW Filter Statistics (tolerance: ±{mw_tolerance} Da):")
        print(f"  Average rank: {np.mean(ranks):.2f}")
        print(f"  Median rank: {np.median(ranks):.2f}")
        print(f"  Top-1 success: {success_rates[0]:.4f}")
        print(f"  Top-5 success: {success_rates[4]:.4f}" if rank_top >= 5 else "")
        
        return {
            'ranks': np.array(ranks),
            'success_rates': np.array(success_rates),
            'rank_top': rank_top,
            'mw_tolerance': mw_tolerance
        }
    
    def spectrum_retrieval_with_mw_filter(
        self,
        query_spectra: np.ndarray,
        database_spectra: np.ndarray,
        query_ids: List[str],
        database_ids: List[str],
        query_mw: List[float],
        database_mw: List[float],
        smiles_list: List[str],
        mw_tolerance: float = 5.0,
        true_indices: np.ndarray = None,
        rank_top: int = 20
    ) -> Dict:
        """
        Complete spectrum retrieval with molecular weight filtering
        
        Args:
            query_spectra: Query spectra
            database_spectra: Database spectra
            query_ids: IDs for query spectra
            database_ids: IDs for database spectra
            query_mw: Molecular weights for query spectra
            database_mw: Molecular weights for database spectra
            smiles_list: SMILES strings
            mw_tolerance: Tolerance for molecular weight matching
            true_indices: True matching indices
            rank_top: Maximum rank for analysis
        
        Returns:
            Dictionary with retrieval results
        """
        print("="*60)
        print("Spectrum Retrieval with MW Filter")
        print("="*60)
        print(f"MW Tolerance: ±{mw_tolerance} Da")
        print(f"Query spectra: {len(query_spectra)}")
        print(f"Database spectra: {len(database_spectra)}")
        
        # Calculate PCC matrix
        print("\nCalculating PCC matrix...")
        pcc_matrix = self.calculate_pcc_matrix(query_spectra, database_spectra)
        
        if true_indices is None:
            true_indices = np.arange(len(query_spectra))
        
        # Perform rank analysis with MW filter
        print("\nPerforming rank analysis with MW filter...")
        rank_results = self.rank_analysis_with_mw_filter(
            pcc_matrix, true_indices, database_mw, mw_tolerance, rank_top
        )
        
        # Get top-1 matches for functional group analysis
        top1_indices = []
        for i in range(len(query_spectra)):
            true_mw = query_mw[i]
            mw_mask = np.abs(np.array(database_mw) - true_mw) <= mw_tolerance
            if np.any(mw_mask):
                filtered_scores = pcc_matrix[i].copy()
                filtered_scores[~mw_mask] = -np.inf
                top1_idx = np.argmax(filtered_scores)
                top1_indices.append(top1_idx)
            else:
                top1_indices.append(np.argmax(pcc_matrix[i]))
        
        # Functional group analysis
        print("\nPerforming functional group analysis...")
        fg_stats = self.functional_group_analysis_with_element_filter(
            smiles_list, top1_indices
        )
        
        # Print results
        print("\n" + "="*60)
        print("Retrieval Results (with MW Filter)")
        print("="*60)
        print(f"\nRank Analysis (Top {rank_top}):")
        for rank in [1, 5, 10, 20]:
            if rank <= rank_top:
                rate = rank_results['success_rates'][rank-1]
                print(f"  Top-{rank} success rate: {rate:.4f} ({rate*100:.1f}%)")
        
        print(f"\nFunctional Group Statistics (Top-1 matches):")
        print(f"{'Functional Group':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-"*61)
        for fg, stats in sorted(fg_stats.items(), key=lambda x: x[1]['f1_score'], reverse=True)[:15]:
            print(f"{fg:<25} {stats['precision']:<12.4f} {stats['recall']:<12.4f} {stats['f1_score']:<12.4f}")
        
        results = {
            'pcc_matrix': pcc_matrix,
            'rank_analysis': rank_results,
            'functional_group_stats': fg_stats,
            'top1_indices': top1_indices
        }
        
        return results

def run_application_analysis():
    """
    Main function to run application analysis
    """
    print("="*60)
    print("Infrared Spectrum Prediction Application Analysis")
    print("="*60)
    
    from utils import load_dict, save_dict
    from train import load_data_from_pkl
    from sklearn.model_selection import train_test_split
    
    try:
        # Load evaluation results
        gpr_eval = load_dict('gpr_evaluation')
        nn_eval = load_dict('nn_evaluation')
        
        # Extract test set predictions and targets
        test_targets = gpr_eval['test']['true']
        test_sim = gpr_eval['test']['sim']  # Theoretical spectra
        test_pred_gpr = gpr_eval['test']['pred']
        test_pred_nn = nn_eval['test']['pred']
        test_ids = gpr_eval['test']['ids']
        
        # Load data.pkl for SMILES
        features, targets, ids, mws, smiles = load_data_from_pkl()
        
        # Split to get test set SMILES
        (_, _, 
         _, _,
         _, _,
         _, _,
         test_smiles, _) = train_test_split(
            features, targets, ids, mws, smiles,
            test_size=0.5, random_state=23
        )
        
        print(f"Test set size: {len(test_ids)}")
        print(f"Test predictions shape: {test_pred_gpr.shape}")
        
        # Initialize analyzer
        analyzer = InfraredSpectrumApplication(use_gpu=True)
        
        # ============================================================
        # Analysis 1: Test set retrieval using GPR predictions as database
        # ============================================================
        print("\n" + "="*60)
        print("Analysis 1: Test Set Retrieval (GPR Predictions as Database)")
        print("="*60)
        
        # Use test targets as queries, GPR predictions as database
        query_spectra = test_targets  # True spectra
        database_spectra_gpr = test_pred_gpr  # GPR predictions
        
        gpr_results = analyzer.spectrum_retrieval_analysis(
            query_spectra=query_spectra,
            database_spectra=database_spectra_gpr,
            query_ids=test_ids,
            database_ids=test_ids,
            smiles_list=test_smiles,
            true_indices=np.arange(len(test_ids)),
            rank_top=20
        )
        
        analyzer.plot_rank_curve(gpr_results['rank_analysis'], save_name="gpr_rank_curve")
        analyzer.plot_functional_group_bar(gpr_results['functional_group_stats'], 
                                          save_name="gpr_fg_performance")
        
        # ============================================================
        # Analysis 2: Test set retrieval using NN predictions as database
        # ============================================================
        print("\n" + "="*60)
        print("Analysis 2: Test Set Retrieval (NN Predictions as Database)")
        print("="*60)
        
        database_spectra_nn = test_pred_nn
        
        nn_results = analyzer.spectrum_retrieval_analysis(
            query_spectra=query_spectra,
            database_spectra=database_spectra_nn,
            query_ids=test_ids,
            database_ids=test_ids,
            smiles_list=test_smiles,
            true_indices=np.arange(len(test_ids)),
            rank_top=20
        )
        
        analyzer.plot_rank_curve(nn_results['rank_analysis'], save_name="nn_rank_curve")
        analyzer.plot_functional_group_bar(nn_results['functional_group_stats'], 
                                          save_name="nn_fg_performance")
        
        # ============================================================
        # Analysis 3: Single spectrum test example
        # ============================================================
        print("\n" + "="*60)
        print("Analysis 3: Single Spectrum Test Example")
        print("="*60)
        
        # Select a test sample
        test_idx = 23
        query_spectrum = test_targets[test_idx]
        query_id = test_ids[test_idx]
        query_smiles = test_smiles[test_idx]
        
        print(f"\nQuery Spectrum:")
        print(f"  ID: {query_id}")
        print(f"  SMILES: {query_smiles}")
        
        # Retrieve using GPR database
        print("\n" + "-"*40)
        print("Retrieval from GPR Predictions Database:")
        print("-"*40)
        
        gpr_single_result = analyzer.single_spectrum_retrieval(
            query_spectrum=query_spectrum,
            database_spectra=test_pred_gpr,
            database_ids=test_ids,
            database_smiles=test_smiles,
            true_id=query_id
        )
        
        print(f"True spectrum rank: {gpr_single_result['rank_of_true']}")
        print("\nTop 5 matches:")
        for match in gpr_single_result['top_matches'][:5]:
            print(f"  Rank {match['rank']}: ID={match['id']}, PCC={match['pcc']:.4f}, SMILES={match['smiles']}")
        
        # Retrieve using NN database
        print("\n" + "-"*40)
        print("Retrieval from NN Predictions Database:")
        print("-"*40)
        
        nn_single_result = analyzer.single_spectrum_retrieval(
            query_spectrum=query_spectrum,
            database_spectra=test_pred_nn,
            database_ids=test_ids,
            database_smiles=test_smiles,
            true_id=query_id
        )
        
        print(f"True spectrum rank: {nn_single_result['rank_of_true']}")
        print("\nTop 5 matches:")
        for match in nn_single_result['top_matches'][:5]:
            print(f"  Rank {match['rank']}: ID={match['id']}, PCC={match['pcc']:.4f}, SMILES={match['smiles']}")
        
        # ============================================================
        # Analysis 4: External database interface example
        # ============================================================
        print("\n" + "="*60)
        print("Analysis 4: External Database Interface")
        print("="*60)
        
        print("\nThis interface allows querying against any external spectral database.")
        print("Example: Using theoretical spectra as external database")
        
        # Example: Use theoretical spectra as external database
        external_database_spectra = test_sim  # Theoretical DFT spectra
        external_database_ids = test_ids
        external_database_smiles = test_smiles
        
        external_result = analyzer.single_spectrum_retrieval(
            query_spectrum=query_spectrum,
            database_spectra=external_database_spectra,
            database_ids=external_database_ids,
            database_smiles=external_database_smiles,
            true_id=query_id
        )
        
        print(f"\nQuerying against Theoretical DFT Database:")
        print(f"True spectrum rank: {external_result['rank_of_true']}")
        print("\nTop 5 matches:")
        for match in external_result['top_matches'][:5]:
            print(f"  Rank {match['rank']}: ID={match['id']}, PCC={match['pcc']:.4f}, SMILES={match['smiles']}")
        
        # ============================================================
        # Analysis 5: Model Comparison Plot
        # ============================================================
        print("\n" + "="*60)
        print("Analysis 5: Model Comparison")
        print("="*60)
        
        fig, ax = plt.subplots(figsize=(8/2.54, 6/2.54))
        
        ranks = np.arange(1, 21)
        ax.step(ranks, gpr_results['rank_analysis']['success_rates'], 
                label='GPR', color='#33ABC1', linewidth=2, where='mid')
        ax.step(ranks, nn_results['rank_analysis']['success_rates'], 
                label='NN', color='#A51C36', linewidth=2, where='mid')
        
        for i in range(1, 11):
            i_s = i / 10
            ax.axhline(i_s, color='gray', alpha=0.3, linestyle='--', linewidth=0.5)
        
        ax.set_xlabel('Rank', fontsize=8)
        ax.set_ylabel('Success Rate', fontsize=8)
        ax.set_xlim(0, 21)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        results_dir = os.path.join(PATH_CONFIG.RESULTS_DIR, 'application')
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, "model_comparison_rank.pdf")
        fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Comparison figure saved to: {save_path}")
        
        plt.show()
        plt.close(fig)
        
        # Save results
        gpr_results_serializable = {
            'rank_analysis': {
                'success_rates': gpr_results['rank_analysis']['success_rates'].tolist(),
                'rank_top': gpr_results['rank_analysis']['rank_top']
            },
            'functional_group_stats': {
                fg: {k: float(v) if isinstance(v, (int, float)) else v 
                     for k, v in stats.items()}
                for fg, stats in gpr_results['functional_group_stats'].items()
            }
        }
        save_dict(gpr_results_serializable, 'gpr_application_results')
        
        nn_results_serializable = {
            'rank_analysis': {
                'success_rates': nn_results['rank_analysis']['success_rates'].tolist(),
                'rank_top': nn_results['rank_analysis']['rank_top']
            },
            'functional_group_stats': {
                fg: {k: float(v) if isinstance(v, (int, float)) else v 
                     for k, v in stats.items()}
                for fg, stats in nn_results['functional_group_stats'].items()
            }
        }
        save_dict(nn_results_serializable, 'nn_application_results')
        
        print("\n" + "="*60)
        print("Application Analysis Completed!")
        print("="*60)
        
        return gpr_results, nn_results
        
    except Exception as e:
        print(f"Error in application analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def run_application_with_filters():
    """
    Run application analysis with element-based filtering and MW constraints
    """
    print("="*60)
    print("Application Analysis with Filters")
    print("="*60)
    
    from utils import load_dict, save_dict
    from train import load_data_from_pkl
    from sklearn.model_selection import train_test_split
    from rdkit.Chem import Descriptors
    
    try:
        # Load evaluation results
        gpr_eval = load_dict('gpr_evaluation')
        nn_eval = load_dict('nn_evaluation')
        
        # Extract test set data
        test_targets = gpr_eval['test']['true']
        test_sim = gpr_eval['test']['sim']
        test_pred_gpr = gpr_eval['test']['pred']
        test_pred_nn = nn_eval['test']['pred']
        test_ids = gpr_eval['test']['ids']
        
        # Load data.pkl for SMILES
        features, targets, ids, mws, smiles = load_data_from_pkl()
        
        # Split to get test set
        (_, _, 
         _, _,
         _, _,
         _, _,
         test_smiles, _) = train_test_split(
            features, targets, ids, mws, smiles,
            test_size=0.5, random_state=23
        )
        
        # Calculate molecular weights using RDKit
        print("\nCalculating molecular weights...")
        test_mw = []
        for smi in test_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                test_mw.append(Descriptors.ExactMolWt(mol))
            else:
                test_mw.append(0.0)
        
        test_mw = np.array(test_mw)
        
        print(f"Test set size: {len(test_ids)}")
        print(f"MW range: {test_mw.min():.1f} - {test_mw.max():.1f} Da")
        
        # Initialize analyzer
        analyzer = InfraredSpectrumApplication(use_gpu=True)
        
        # ============================================================
        # Analysis 1: Element-based classification
        # ============================================================
        print("\n" + "="*60)
        print("Analysis 1: Element-based Classification")
        print("="*60)
        
        classification = analyzer.classify_molecules_by_elements(
            test_ids, test_smiles, test_targets
        )
        
        # ============================================================
        # Analysis 2: Functional group analysis with element filtering
        # ============================================================
        print("\n" + "="*60)
        print("Analysis 2: Functional Group Analysis with Element Filtering")
        print("="*60)
        
        # Get top-1 matches from GPR
        pcc_matrix = analyzer.calculate_pcc_matrix(test_targets, test_pred_gpr)
        top1_indices = np.argmax(pcc_matrix, axis=1)
        
        # Run filtered functional group analysis
        fg_stats_filtered = analyzer.functional_group_analysis_with_element_filter(
            test_smiles, top1_indices, classification
        )
        
        # Plot filtered functional group bar chart
        analyzer.plot_functional_group_bar(
            fg_stats_filtered, 
            top_n=10, 
            save_name="fg_performance_with_element_filter"
        )
        
        # ============================================================
        # Analysis 3: Spectrum retrieval with MW filter (GPR)
        # ============================================================
        print("\n" + "="*60)
        print("Analysis 3: Spectrum Retrieval with MW Filter (GPR)")
        print("="*60)
        
        gpr_mw_results = analyzer.spectrum_retrieval_with_mw_filter(
            query_spectra=test_targets,
            database_spectra=test_pred_gpr,
            query_ids=test_ids,
            database_ids=test_ids,
            query_mw=test_mw,
            database_mw=test_mw,
            smiles_list=test_smiles,
            mw_tolerance=5.0,
            true_indices=np.arange(len(test_ids)),
            rank_top=20
        )
        
        analyzer.plot_rank_curve(gpr_mw_results['rank_analysis'], save_name="gpr_rank_curve_with_mw_filter")
        
        # ============================================================
        # Analysis 4: Spectrum retrieval with MW filter (NN)
        # ============================================================
        print("\n" + "="*60)
        print("Analysis 4: Spectrum Retrieval with MW Filter (NN)")
        print("="*60)
        
        nn_mw_results = analyzer.spectrum_retrieval_with_mw_filter(
            query_spectra=test_targets,
            database_spectra=test_pred_nn,
            query_ids=test_ids,
            database_ids=test_ids,
            query_mw=test_mw,
            database_mw=test_mw,
            smiles_list=test_smiles,
            mw_tolerance=5.0,
            true_indices=np.arange(len(test_ids)),
            rank_top=20
        )
        
        analyzer.plot_rank_curve(nn_mw_results['rank_analysis'], save_name="nn_rank_curve_with_mw_filter")
        
        # ============================================================
        # Analysis 5: Comparison with different MW tolerances
        # ============================================================
        print("\n" + "="*60)
        print("Analysis 5: MW Tolerance Comparison")
        print("="*60)
        
        tolerances = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        top1_rates = []
        
        for tol in tolerances:
            rank_results = analyzer.rank_analysis_with_mw_filter(
                pcc_matrix, np.arange(len(test_ids)), test_mw, tol, 20
            )
            top1_rates.append(rank_results['success_rates'][0])
            print(f"Tolerance ±{tol} Da: Top-1 rate = {rank_results['success_rates'][0]:.4f}")
        
        # Plot MW tolerance comparison
        fig, ax = plt.subplots(figsize=(8/2.54, 6/2.54))
        ax.plot(tolerances, top1_rates, 'o-', color='#33ABC1', linewidth=1.5, markersize=4)
        ax.set_xlabel('MW Tolerance (± Da)', fontsize=8)
        ax.set_ylabel('Top-1 Success Rate', fontsize=8)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        results_dir = os.path.join(PATH_CONFIG.RESULTS_DIR, 'application')
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, "mw_tolerance_comparison.pdf")
        fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
        plt.show()
        plt.close(fig)
        
        # Save results
        save_dict(gpr_mw_results, 'gpr_application_results_with_mw_filter')
        save_dict(nn_mw_results, 'nn_application_results_with_mw_filter')
        
        print("\n" + "="*60)
        print("Application Analysis with Filters Completed!")
        print("="*60)
        
        return gpr_mw_results, nn_mw_results
        
    except Exception as e:
        print(f"Error in application analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None

from sklearn.metrics import roc_curve, auc

def extract_top_elements(matrix: np.ndarray, k: int = 500, ex: str = 'max') -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract top k elements (excluding diagonal) from each row of a matrix
    
    Args:
        matrix: Input matrix (2D numpy array)
        k: Number of elements to extract
        ex: 'max' for maximum values, 'min' for minimum values
    
    Returns:
        Tuple of (indices, values) for top k elements per row
    """
    modified_matrix = matrix.copy()
    
    # Set diagonal to infinity to exclude from selection
    np.fill_diagonal(modified_matrix, -np.inf if ex == 'max' else np.inf)
    
    if ex == 'max':
        sorted_indices = np.argsort(-modified_matrix, axis=1)[:, :k]
    elif ex == 'min':
        sorted_indices = np.argsort(modified_matrix, axis=1)[:, :k]
    else:
        raise ValueError("ex must be 'max' or 'min'")
    
    rows = np.arange(matrix.shape[0]).reshape(-1, 1)
    values = matrix[rows, sorted_indices]
    
    return sorted_indices, values

def plot_roc_by_similarity(iden_matrix: np.ndarray, indices: np.ndarray, 
                          ai_model: str = 'Pred', plot_s: bool = False) -> float:
    """
    Calculate ROC AUC from similarity matrix and selected indices
    
    Args:
        iden_matrix: Similarity matrix (PCC matrix)
        indices: Selected indices for negative samples
        ai_model: Model name for labeling
        plot_s: Whether to plot ROC curve
    
    Returns:
        ROC AUC score
    """
    iden_matrix = np.array(iden_matrix)
    
    predicted_scores = []
    true_labels = []
    
    for i in range(len(iden_matrix)):
        row = iden_matrix[i]
        
        # Positive sample (diagonal)
        pcc_positive = row[i]
        predicted_scores.append(pcc_positive)
        true_labels.append(1)
        
        # Negative samples (selected indices)
        for c_i in indices[i]:
            value = row[c_i]
            predicted_scores.append(value)
            true_labels.append(0)
    
    true_labels_np = np.array(true_labels)
    predicted_scores_np = np.array(predicted_scores)
    
    fpr, tpr, _ = roc_curve(true_labels_np, predicted_scores_np, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    if plot_s:
        print(f"AUC value: {roc_auc:.4f}")
        plot_roc_curve(fpr, tpr, roc_auc, ai_model)
    
    return roc_auc

def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, ai_model: str = 'Pred'):
    """
    Plot individual ROC curve
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        roc_auc: AUC score
        ai_model: Model name
    """
    plt.figure(figsize=(10, 6), dpi=600)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})\n(AI: {ai_model})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title(f'ROC Curve: {ai_model} vs. Experimental Spectra', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_combined_roc(roc_data: Dict, width_cm: float = 8, height_cm: float = 5, 
                      save_name: str = "roc_combined"):
    """
    Plot combined ROC vs similarity analysis
    
    Args:
        roc_data: Dictionary containing similarity and ROC data
        width_cm: Figure width in centimeters
        height_cm: Figure height in centimeters
        save_name: Name for saved figure
    """ 
    # Convert cm to inches
    width_inch = width_cm / 2.54
    height_inch = height_cm / 2.54
    fig, ax = plt.subplots(figsize=(width_inch, height_inch))
    
    # Plot data points
    ax.plot(roc_data['quasi'], roc_data['roc_quasi'], 'o-', 
            markersize=3, label='Predicted (GPR/NN)', 
            color='#A51C36', linewidth=0.8, markeredgecolor='none')
    ax.plot(roc_data['dft'], roc_data['roc_dft'], 's--', 
            markersize=3, label='Theoretical (DFT)', 
            color='#7ABBDB', linewidth=0.8, markeredgecolor='none')
    
    # Set labels
    ax.set_xlabel('Average Maximum Similarity', fontsize=8)
    ax.set_ylabel('AUC Score', fontsize=8)
    
    # Set ticks and limits
    ax.set_xlim(left=0.3, right=1.0)
    ax.set_ylim(bottom=0.5, top=1.0)
    ax.set_xticks(np.arange(0.3, 1.05, 0.1))
    ax.set_yticks(np.arange(0.5, 1.05, 0.1))
    ax.tick_params(axis='both', labelsize=7, width=0.5)
    
    # Set spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    
    # Legend
    legend = ax.legend(loc='lower right',
                      fontsize=7,
                      frameon=True,
                      fancybox=False,
                      edgecolor='black',
                      facecolor='white',
                      framealpha=0.9,
                      borderpad=0.5)
    legend.get_frame().set_linewidth(0.5)
    
    # Grid
    ax.grid(True, alpha=0.2, linewidth=0.3, linestyle='--')
    
    # Save figure
    results_dir = os.path.join(PATH_CONFIG.RESULTS_DIR, 'application')
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"{save_name}.pdf")
    fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    
    plt.show()
    plt.close(fig)

def roc_similarity_analysis(theoretical_data: np.ndarray, 
                           target_data: np.ndarray,
                           predicted_data: np.ndarray,
                           database_sizes: List[int] = None,
                           use_gpu: bool = True) -> Dict:
    """
    Perform ROC similarity analysis comparing theoretical and predicted spectra
    
    This analysis evaluates how well the similarity matrix can distinguish
    correct matches from incorrect ones, as a function of database size.
    
    Args:
        theoretical_data: Theoretical spectra 
        target_data: Target spectra 
        predicted_data: Predicted spectra from GPR or NN
        database_sizes: List of database sizes to evaluate (default: [1,5,10,20,50,100,200,500,1000,2000])
        use_gpu: Whether to use GPU for calculations
    
    Returns:
        Dictionary with similarity and ROC results
    """
    if database_sizes is None:
        database_sizes = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    
    print("="*60)
    print("ROC Similarity Analysis")
    print("="*60)
    print(f"Total database size: {len(target_data)}")
    print(f"Evaluating at database sizes: {database_sizes}")
    
    # Calculate similarity matrices
    print("\nCalculating similarity matrices...")
    gpu_calc = GPUCalculator() if use_gpu else None
    
    if use_gpu:
        # Theoretical vs Target
        iden_matrix_dft = gpu_calc.pearson_matrix_gpu(theoretical_data, target_data)
        # Predicted vs Target
        iden_matrix_pred = gpu_calc.pearson_matrix_gpu(predicted_data, target_data)
    else:
        # CPU calculation
        n_samples = len(target_data)
        iden_matrix_dft = np.zeros((n_samples, n_samples))
        iden_matrix_pred = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                corr_dft = np.corrcoef(theoretical_data[i], target_data[j])[0, 1]
                corr_pred = np.corrcoef(predicted_data[i], target_data[j])[0, 1]
                iden_matrix_dft[i, j] = corr_dft if not np.isnan(corr_dft) else 0.0
                iden_matrix_pred[i, j] = corr_pred if not np.isnan(corr_pred) else 0.0
    
    print(f"Similarity matrices computed: {iden_matrix_dft.shape}")
    
    # Initialize results
    roc_results = {
        'dft': {'similarity': [], 'auc': []},
        'pred': {'similarity': [], 'auc': []},
        'database_sizes': database_sizes
    }
    
    # Analyze at different database sizes
    for k in database_sizes:
        print(f"\nAnalyzing database size: {k}")
        
        # Limit k to actual database size
        actual_k = min(k, len(target_data))
        
        # Extract top k elements for each row
        top_indices_dft, top_values_dft = extract_top_elements(iden_matrix_dft, actual_k, ex='max')
        top_indices_pred, top_values_pred = extract_top_elements(iden_matrix_pred, actual_k, ex='max')
        
        # Calculate average similarity of top matches
        avg_sim_dft = np.mean([np.mean(values) for values in top_values_dft])
        avg_sim_pred = np.mean([np.mean(values) for values in top_values_pred])
        
        # Calculate ROC AUC
        auc_dft = plot_roc_by_similarity(iden_matrix_dft, top_indices_dft, ai_model='DFT', plot_s=False)
        auc_pred = plot_roc_by_similarity(iden_matrix_pred, top_indices_pred, ai_model='Pred', plot_s=False)
        
        roc_results['dft']['similarity'].append(avg_sim_dft)
        roc_results['dft']['auc'].append(auc_dft)
        roc_results['pred']['similarity'].append(avg_sim_pred)
        roc_results['pred']['auc'].append(auc_pred)
        
        print(f"  DFT - Avg Similarity: {avg_sim_dft:.4f}, AUC: {auc_dft:.4f}")
        print(f"  Pred - Avg Similarity: {avg_sim_pred:.4f}, AUC: {auc_pred:.4f}")
    
    # Prepare combined data for plotting
    combined_data = {
        'dft': roc_results['dft']['similarity'],
        'roc_dft': roc_results['dft']['auc'],
        'quasi': roc_results['pred']['similarity'],
        'roc_quasi': roc_results['pred']['auc']
    }
    
    # Plot combined results
    plot_combined_roc(combined_data, save_name="roc_similarity_analysis")
    
    return roc_results

def run_roc_similarity_analysis():
    """
    Main function to run ROC similarity analysis using data from data.pkl
    """
    print("="*60)
    print("ROC Similarity Analysis Application")
    print("="*60)
    
    try:
        # Load data from data.pkl
        from train import load_data_from_pkl
        features, targets, ids, mws, smiles = load_data_from_pkl()
        
        print(f"Loaded data: {len(ids)} samples")
        print(f"Theoretical spectra (features) shape: {features.shape}")
        print(f"Target spectra (targets) shape: {targets.shape}")
        
        # Use targets as experimental data
        exp_data = targets
        
        # Theoretical data (DFT sim_list)
        theoretical_data = features
        
        # For predicted data, we need GPR and NN predictions
        # Load evaluation results
        try:
            gpr_eval = load_dict('gpr_evaluation')
            nn_eval = load_dict('nn_evaluation')
            
            # Get test set indices
            from sklearn.model_selection import train_test_split
            _, test_indices = train_test_split(
                range(len(ids)), test_size=0.5, random_state=23
            )
            
            # Extract test set data
            test_theoretical = features[test_indices]
            test_target = targets[test_indices]
            
            # Get predictions (these should align with test set)
            gpr_pred = gpr_eval['test']['pred']
            nn_pred = nn_eval['test']['pred']
            
            print(f"\nTest set size: {len(test_target)}")
            
            # Run ROC similarity analysis for GPR
            print("\n" + "="*60)
            print("GPR Model ROC Similarity Analysis")
            print("="*60)
            
            gpr_roc_results = roc_similarity_analysis(
                theoretical_data=test_theoretical,
                target_data=test_target,
                predicted_data=gpr_pred.reshape(-1, 1) if len(gpr_pred.shape) == 1 else gpr_pred,
                database_sizes=[1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
                use_gpu=True
            )
            
            # Save GPR results
            save_dict(gpr_roc_results, 'gpr_roc_similarity_results')
            
            # Run ROC similarity analysis for NN
            print("\n" + "="*60)
            print("NN Model ROC Similarity Analysis")
            print("="*60)
            
            nn_roc_results = roc_similarity_analysis(
                theoretical_data=test_theoretical,
                target_data=test_target,
                predicted_data=nn_pred.reshape(-1, 1) if len(nn_pred.shape) == 1 else nn_pred,
                database_sizes=[1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
                use_gpu=True
            )
            
            # Save NN results
            save_dict(nn_roc_results, 'nn_roc_similarity_results')
            
            # Compare GPR and NN
            print("\n" + "="*60)
            print("Model Comparison")
            print("="*60)
            
            # Create comparison plot
            width_inch = 8 / 2.54
            height_inch = 6 / 2.54
            fig, ax = plt.subplots(figsize=(width_inch, height_inch))
            
            ax.plot(gpr_roc_results['pred']['similarity'], gpr_roc_results['pred']['auc'], 
                   'o-', markersize=3, label='GPR', color='#A51C36', linewidth=0.8)
            ax.plot(nn_roc_results['pred']['similarity'], nn_roc_results['pred']['auc'], 
                   's-', markersize=3, label='NN', color='#7ABBDB', linewidth=0.8)
            ax.plot(gpr_roc_results['dft']['similarity'], gpr_roc_results['dft']['auc'], 
                   'o--', markersize=3, label='Theoretical (DFT)', color='#84BA42', linewidth=0.8)
            
            ax.set_xlabel('Average Maximum Similarity', fontsize=8)
            ax.set_ylabel('AUC Score', fontsize=8)
            ax.set_xlim(left=0.3, right=1.0)
            ax.set_ylim(bottom=0.5, top=1.0)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            
            plt.tight_layout()
            
            results_dir = os.path.join(PATH_CONFIG.RESULTS_DIR, 'application')
            os.makedirs(results_dir, exist_ok=True)
            save_path = os.path.join(results_dir, "model_comparison_roc_similarity.pdf")
            fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
            print(f"Comparison figure saved to: {save_path}")
            
            plt.show()
            plt.close(fig)
            
            return gpr_roc_results, nn_roc_results
            
        except Exception as e:
            print(f"Error loading evaluation results: {e}")
            print("Running analysis on full dataset (without train/test split)...")
            
            # Run on full dataset
            roc_results = roc_similarity_analysis(
                theoretical_data=theoretical_data,
                target_data=exp_data,
                predicted_data=theoretical_data,  # Placeholder
                database_sizes=[1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
                use_gpu=True
            )
            
            save_dict(roc_results, 'roc_similarity_results')
            return roc_results, None
            
    except Exception as e:
        print(f"Error in ROC similarity analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def run_all_applications():
    """
    Run all application analyses
    """
    print("\n" + "="*60)
    print("Running All Application Analyses")
    print("="*60)
    
    # Run spectrum retrieval analysis
    run_application_analysis()
    run_application_with_filters()    
    
    # Run ROC similarity analysis
    gpr_roc, nn_roc = run_roc_similarity_analysis()

if __name__ == "__main__":
    run_all_applications()