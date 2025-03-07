import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler

def create_hetero_data(patient_df, modalities, gene_data=None, radiomics_data=None, 
                       structure_file='data/structure_distance_clean.csv',
                       co_expression_file='data/co_expression.csv',
                       dwi_matrix_file='data/DWI_Matrix.csv',
                       struct=True, coexp=True, device='cpu', binary=True):
    """
    Create a heterogeneous graph data object from patient data.
    
    Args:
        patient_df: Pandas DataFrame containing patient information
        modalities: List of modalities to use (G for gene, R for radiomics)
        gene_data: Path to gene data CSV file or DataFrame (required if 'G' in modalities)
        radiomics_data: Path to radiomics data CSV file or DataFrame (required if 'R' in modalities)
        structure_file: Path to structure distance file
        co_expression_file: Path to co-expression file
        dwi_matrix_file: Path to DWI matrix file
        struct: Whether to include structure-structure connections
        coexp: Whether to include gene co-expression connections
        device: Device to put the data on
        binary: Whether to use binary classification (AD vs CTL) or multiclass
        
    Returns:
        HeteroData object for model training/inference
    """
    # List of relevant gene names
    gene_mapping = [
        'COL25A1', 'APBA1', 'APBA2', 'APBA3', 'APBB1', 'APP', 'CDK5R1', 'LRP1', 
        'NCSTN', 'UQCRC1', 'ACE', 'ADAM19', 'AGER', 'ANKS1B', 'APBB2', 'APH1A', 
        'APLP2', 'APPBP2', 'BACE2', 'BDNF', 'BPTF', 'C5AR1', 'CASP2', 'CASP3', 
        'CAT', 'CDK5RAP2', 'CIB1', 'CLSTN1', 'CR1', 'CRH', 'CTSB', 'CTSD', 'DCHS2', 
        'DHCR24', 'DNM1L', 'DNMBP', 'DPYSL2', 'EXOC3L2', 'FRMD4A', 'GLRX', 'GPR3', 
        'IL1A', 'KIDINS220', 'MAPT', 'MARK4', 'MSR1', 'MT3', 'NAE1', 'NDUFB8', 
        'NQO1', 'OLR1', 'PCMT1', 'PICALM', 'PITRM1', 'PLAU', 'PLD3', 'PSEN2', 
        'PSENEN', 'PZP', 'QPRT', 'RBM45', 'RCAN1', 'SAA1', 'SAA2', 'SLC30A6', 
        'SLC8A2', 'SNAP91', 'SNCA', 'SNRNP70', 'SORCS3', 'TFAM', 'THBS4', 'TMED10', 
        'TNF', 'UBQLN1'
    ]
    
    hetero_data = HeteroData()

    # Step 1: Add patient nodes
    patient_df['APOE_encoded'] = patient_df['APOE'].apply(lambda x: 1 if x == 'E3E3' else 2 if x == 'E3E4' else 3 if x == 'E4E4' else 4 if x == 'E2E3' else 0)
    patient_df['Sex'] = patient_df['Sex'].apply(lambda x: 1 if x == 'Male' else 0)
    hetero_data['patient'].x = torch.tensor(patient_df[['APOE_encoded', 'Sex', 'MMSE']].values, dtype=torch.float)
    
    if binary:
        hetero_data['patient'].y = torch.tensor(patient_df['Subtype'].apply(lambda x: 0 if x == 'CTL' else 1).values, dtype=torch.long)
    else:
        hetero_data['patient'].y = torch.tensor(patient_df['Subtype'].apply(lambda x: 0 if x == 'CTL' else 1 if x == 'MCI' else 2 if x == 'AD' else 3).values, dtype=torch.long)
    
    # Step 2: Add gene nodes and patient-gene edges if applicable
    if 'G' in modalities:
        if gene_data is None:
            raise ValueError("Gene data must be provided when using G modality")
            
        if isinstance(gene_data, str):
            gene_df = pd.read_csv(gene_data)
        else:
            gene_df = gene_data
            
        gene_df = gene_df[gene_df['Key'].isin(patient_df['Key'])]
        
        num_patients = len(patient_df)
        num_genes = gene_df.shape[1] - 1
        gene_expression_values = gene_df.drop(columns=['Key']).values.flatten()
        hetero_data['gene'].x = torch.tensor(gene_expression_values, dtype=torch.float).unsqueeze(1)

        patient_indices = torch.arange(num_patients).repeat_interleave(num_genes)
        gene_indices = torch.arange(num_genes).repeat(num_patients)
        patient_gene_edges = torch.stack([patient_indices, gene_indices], dim=0)
        hetero_data['patient', 'associated_with', 'gene'].edge_index = patient_gene_edges

        if coexp:
            co_expression_df = pd.read_csv(co_expression_file)
            gene_to_idx = {gene: idx for idx, gene in enumerate(gene_mapping)}
            gene_gene_edges = []
            gene_gene_weights = []

            for _, row in co_expression_df.iterrows():
                gene1, gene2, weight = row['Gene1'], row['Gene2'], row['Weight']
                if gene1 in gene_to_idx and gene2 in gene_to_idx:
                    gene1_idx = gene_to_idx[gene1]
                    gene2_idx = gene_to_idx[gene2]
                    gene_gene_edges.append([gene1_idx, gene2_idx])
                    gene_gene_weights.append(weight)

            gene_gene_edges = torch.tensor(gene_gene_edges, dtype=torch.long).t()
            gene_gene_weights = torch.tensor(gene_gene_weights, dtype=torch.float)
            hetero_data['gene', 'co_expressed_with', 'gene'].edge_index = gene_gene_edges
            hetero_data['gene', 'co_expressed_with', 'gene'].edge_attr = gene_gene_weights.unsqueeze(1)

    if 'R' in modalities:
        if radiomics_data is None:
            raise ValueError("Radiomics data must be provided when using R modality")
            
        if isinstance(radiomics_data, str):
            radiomics_df = pd.read_csv(radiomics_data)
        else:
            radiomics_df = radiomics_data
            
        radiomics_df['PatientID'] = radiomics_df['patient_structure'].apply(lambda x: x.split('_')[0])
        radiomics_df = radiomics_df[radiomics_df['PatientID'].isin(patient_df['Key'])]
        radiomics_features = [col for col in radiomics_df.columns if 'original_' in col]
        scaler = StandardScaler()
        radiomics_df[radiomics_features] = scaler.fit_transform(radiomics_df[radiomics_features])
        hetero_data['structure'].x = torch.tensor(radiomics_df[radiomics_features].values, dtype=torch.float)
        
        patient_to_structure_edges = radiomics_df[['patient_structure', 'PatientID']].drop_duplicates()
        patient_map = {pid: idx for idx, pid in enumerate(patient_df['Key'].unique())}
        structure_map = {structure: idx for idx, structure in enumerate(radiomics_df['patient_structure'])}
        
        patient_edges = patient_to_structure_edges['PatientID'].map(patient_map).values
        structure_edges = patient_to_structure_edges['patient_structure'].map(structure_map).values
        hetero_data['patient', 'interacts', 'structure'].edge_index = torch.tensor([patient_edges, structure_edges], dtype=torch.long)

        if struct:
            # Read distance and SWI connectivity data
            structure_dist_df = pd.read_csv(structure_file)
            swi_matrix = pd.read_csv(dwi_matrix_file, header=None).values
            
            # Filter distance data for relevant patients
            patient_ids = set(patient_df['Key'].unique())
            structure_dist_df = structure_dist_df[structure_dist_df['structure_1'].apply(lambda x: x.split('_')[0] in patient_ids)]
            
            # Create a mapping of unique structure names to indices
            unique_structures = pd.concat([
                structure_dist_df['structure_1'].apply(lambda x: x.split('_')[1]),
                structure_dist_df['structure_2'].apply(lambda x: x.split('_')[1])
            ]).unique()
            structure_to_idx = {name: idx for idx, name in enumerate(sorted(unique_structures))}
            
            # Scale distance values (inverse since shorter distance means stronger connection)
            scaler = StandardScaler()
            distances = structure_dist_df['distance'].values.reshape(-1, 1)
            scaled_distances = scaler.fit_transform(distances)
            scaled_distances = 1 / (1 + np.exp(scaled_distances))  # Sigmoid transformation to [0,1]
            
            # Get SWI connectivity values using the structure indices
            swi_weights = []
            for _, row in structure_dist_df.iterrows():
                struct1_name = row['structure_1'].split('_')[1]
                struct2_name = row['structure_2'].split('_')[1]
                struct1_idx = structure_to_idx[struct1_name]
                struct2_idx = structure_to_idx[struct2_name]
                swi_weight = swi_matrix[struct1_idx, struct2_idx]
                swi_weights.append(swi_weight)
            
            # Scale SWI connectivity values
            swi_weights = np.array(swi_weights).reshape(-1, 1)
            scaled_swi = scaler.fit_transform(swi_weights)
            scaled_swi = 1 / (1 + np.exp(scaled_swi))  # Sigmoid transformation to [0,1]
            
            # Combine the two sources of edge weights
            a = 0.5  # weight for distance-based edges
            b = 0.5  # weight for SWI connectivity-based edges
            combined_weights = a * scaled_distances + b * scaled_swi
            
            # Create edge index and attributes
            structure_1_edges = structure_dist_df['structure_1'].map(structure_map).values
            structure_2_edges = structure_dist_df['structure_2'].map(structure_map).values
            hetero_data['structure', 'related_to', 'structure'].edge_index = torch.tensor([structure_1_edges, structure_2_edges], dtype=torch.long)
            hetero_data['structure', 'related_to', 'structure'].edge_attr = torch.tensor(combined_weights, dtype=torch.float)

    return hetero_data.to(device)