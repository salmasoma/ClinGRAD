import torch
import pandas as pd
import numpy as np
import argparse
from torch_geometric.loader import DataLoader

# Import custom modules
from model import ClinGRAD
from data_utils import create_hetero_data

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ClinGRAD Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--patient_data', type=str, required=True, help='Path to patient data CSV')
    parser.add_argument('--gene_data', type=str, help='Path to gene data CSV (optional if not using G modality)')
    parser.add_argument('--radiomics_data', type=str, help='Path to radiomics data CSV (optional if not using R modality)')
    parser.add_argument('--structure_distance_file', type=str, default='data/structure_distance_clean.csv', 
                        help='Path to structure distance file')
    parser.add_argument('--co_expression_file', type=str, default='data/co_expression.csv',
                        help='Path to co-expression file')
    parser.add_argument('--dwi_matrix_file', type=str, default='data/DWI_Matrix.csv',
                        help='Path to DWI connectivity matrix file')
    parser.add_argument('--output_path', type=str, default='predictions.csv', help='Path to save predictions')
    parser.add_argument('--modality', type=str, default='RG', help='Modalities to use (R=radiomics, G=gene)')
    parser.add_argument('--struct', action='store_true', default=True, help='Include structure-structure connections')
    parser.add_argument('--coexp', action='store_true', default=True, help='Include gene coexpression connections')
    parser.add_argument('--binary', action='store_true', default=True, help='Binary or multiclass classification')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
    
    return parser.parse_args()

def run_inference(args):
    """
    Run inference using a pre-trained ClinGRAD model
    
    Args:
        args: Command line arguments
    
    Returns:
        DataFrame with predictions
    """
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Split modalities
    modalities = list(args.modality)
    print(f"Using modalities: {modalities}")
    
    # Constants
    NUM_CLASSES = 2 if args.binary else 4
    
    # Load patient data
    print("Loading patient data...")
    patient_df = pd.read_csv(args.patient_data)
    
    # Create hetero data
    print("Creating heterogeneous graph data...")
    data = create_hetero_data(
        patient_df=patient_df, 
        modalities=modalities,
        gene_data=args.gene_data if 'G' in modalities else None,
        radiomics_data=args.radiomics_data if 'R' in modalities else None,
        structure_file=args.structure_distance_file,
        co_expression_file=args.co_expression_file,
        dwi_matrix_file=args.dwi_matrix_file,
        struct=args.struct,
        coexp=args.coexp,
        device=device
    )
    
    # Create data loader
    loader = DataLoader([data], batch_size=1, shuffle=False)
    
    # Load pre-trained model
    print(f"Loading model from {args.model_path}...")
    model = ClinGRAD(
        hidden_channels=64, 
        out_channels=NUM_CLASSES, 
        metadata=data.metadata(), 
        modalities=modalities,
        struct=args.struct,
        coexp=args.coexp,
        heads=args.heads
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Run inference
    print("Running inference...")
    all_probabilities = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            probabilities = torch.nn.functional.softmax(out, dim=1)
            predictions = out.argmax(dim=1)
            
            all_probabilities.append(probabilities.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
    
    # Concatenate results from batches
    all_probabilities = np.concatenate(all_probabilities, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    # Map predictions to class labels
    if args.binary:
        class_mapping = {0: 'CTL', 1: 'AD'}
    else:
        class_mapping = {0: 'CTL', 1: 'MCI', 2: 'AD', 3: 'VaD'}
    
    # Create results dataframe
    results = pd.DataFrame({
        'PatientID': patient_df['Key'],
        'Predicted_Class_Idx': all_predictions,
        'Predicted_Class': [class_mapping[idx] for idx in all_predictions]
    })
    
    # Add probability columns
    for i in range(NUM_CLASSES):
        class_label = class_mapping[i]
        results[f'Probability_{class_label}'] = all_probabilities[:, i]
    
    # Save results
    results.to_csv(args.output_path, index=False)
    print(f"Inference complete! Results saved to {args.output_path}")
    
    # Display summary
    print("\nPrediction Summary:")
    print(results['Predicted_Class'].value_counts())
    
    return results

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)