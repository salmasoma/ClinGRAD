import torch
from torch_geometric.nn import HeteroConv, GATConv, Linear

class ClinGRAD(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, modalities, struct=True, coexp=True, heads=4):
        """
        Initialize ClinGRAD model
        
        Args:
            hidden_channels (int): Number of hidden channels
            out_channels (int): Number of output classes
            metadata (dict): Metadata from the HeteroData object
            modalities (list): List of modalities to use (G for gene, R for radiomics)
            struct (bool): Whether to include structure-structure connections
            coexp (bool): Whether to include gene coexpression connections
            heads (int): Number of attention heads
        """
        super().__init__()
        self.patient_lin = Linear(4, hidden_channels)
        self.gene_lin = Linear(1, hidden_channels)
        self.structure_lin = Linear(107, hidden_channels)
        self.modalities = modalities
        self.struct = struct
        self.coexp = coexp
        self.heads = heads
        
        if 'G' in modalities and 'R' in modalities:
            if coexp:
                if struct:
                    self.conv1 = HeteroConv({
                        ('patient', 'associated_with', 'gene'): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                        ('gene', 'rev_associated_with', 'patient'): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                        ('gene', 'co_expressed_with', 'gene'): GATConv(hidden_channels, hidden_channels, heads=heads, edge_dim=1, add_self_loops=False),
                        ('patient', 'interacts', 'structure'): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                        ('structure', 'rev_interacts', 'patient'): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                        ('structure', 'related_to', 'structure'): GATConv(hidden_channels, hidden_channels, heads=heads, edge_dim=1, add_self_loops=False),
                    })
                    
                    self.conv2 = HeteroConv({
                        ('patient', 'associated_with', 'gene'): GATConv((hidden_channels * heads, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                        ('gene', 'rev_associated_with', 'patient'): GATConv((hidden_channels * heads, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                        ('gene', 'co_expressed_with', 'gene'): GATConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=1, add_self_loops=False),
                        ('patient', 'interacts', 'structure'): GATConv((hidden_channels * heads, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                        ('structure', 'rev_interacts', 'patient'): GATConv((hidden_channels * heads, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                        ('structure', 'related_to', 'structure'): GATConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=1, add_self_loops=False),
                    })
                else:
                    self.conv1 = HeteroConv({
                        ('patient', 'associated_with', 'gene'): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                        ('gene', 'rev_associated_with', 'patient'): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                        ('gene', 'co_expressed_with', 'gene'): GATConv(hidden_channels, hidden_channels, heads=heads, edge_dim=1, add_self_loops=False),
                        ('patient', 'interacts', 'structure'): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                        ('structure', 'rev_interacts', 'patient'): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                    })
                    
                    self.conv2 = HeteroConv({
                        ('patient', 'associated_with', 'gene'): GATConv((hidden_channels * heads, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                        ('gene', 'rev_associated_with', 'patient'): GATConv((hidden_channels * heads, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                        ('gene', 'co_expressed_with', 'gene'): GATConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=1, add_self_loops=False),
                        ('patient', 'interacts', 'structure'): GATConv((hidden_channels * heads, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                        ('structure', 'rev_interacts', 'patient'): GATConv((hidden_channels * heads, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                    })
            else:
                if struct:
                    self.conv1 = HeteroConv({
                        ('patient', 'associated_with', 'gene'): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                        ('gene', 'rev_associated_with', 'patient'): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                        ('patient', 'interacts', 'structure'): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                        ('structure', 'rev_interacts', 'patient'): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                        ('structure', 'related_to', 'structure'): GATConv(hidden_channels, hidden_channels, heads=heads, edge_dim=1, add_self_loops=False),
                    })
                    
                    self.conv2 = HeteroConv({
                        ('patient', 'associated_with', 'gene'): GATConv((hidden_channels * heads, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                        ('gene', 'rev_associated_with', 'patient'): GATConv((hidden_channels * heads, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                        ('patient', 'interacts', 'structure'): GATConv((hidden_channels * heads, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                        ('structure', 'rev_interacts', 'patient'): GATConv((hidden_channels * heads, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                        ('structure', 'related_to', 'structure'): GATConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=1, add_self_loops=False),
                    })

        elif 'G' in modalities:
            if coexp:
                self.conv1 = HeteroConv({
                    ('patient', 'associated_with', 'gene'): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                    ('gene', 'rev_associated_with', 'patient'): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                    ('gene', 'co_expressed_with', 'gene'): GATConv(hidden_channels, hidden_channels, heads=heads, edge_dim=1, add_self_loops=False),
                })
                    
                self.conv2 = HeteroConv({
                    ('patient', 'associated_with', 'gene'): GATConv((hidden_channels * heads, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                    ('gene', 'rev_associated_with', 'patient'): GATConv((hidden_channels * heads, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                    ('gene', 'co_expressed_with', 'gene'): GATConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=1, add_self_loops=False),
                })
            else:
                self.conv1 = HeteroConv({
                    ('patient', 'associated_with', 'gene'): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                    ('gene', 'rev_associated_with', 'patient'): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                })
                    
                self.conv2 = HeteroConv({
                    ('patient', 'associated_with', 'gene'): GATConv((hidden_channels * heads, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                    ('gene', 'rev_associated_with', 'patient'): GATConv((hidden_channels * heads, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                })

        elif 'R' in modalities:
            if struct:
                self.conv1 = HeteroConv({
                    ('patient', 'interacts', 'structure'): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                    ('structure', 'rev_interacts', 'patient'): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                    ('structure', 'related_to', 'structure'): GATConv(hidden_channels, hidden_channels, heads=heads, edge_dim=1, add_self_loops=False),
                })
                    
                self.conv2 = HeteroConv({
                    ('patient', 'interacts', 'structure'): GATConv((hidden_channels * heads, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                    ('structure', 'rev_interacts', 'patient'): GATConv((hidden_channels * heads, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                    ('structure', 'related_to', 'structure'): GATConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=1, add_self_loops=False),
                })
            else:
                self.conv1 = HeteroConv({
                    ('patient', 'interacts', 'structure'): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                    ('structure', 'rev_interacts', 'patient'): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                })
                    
                self.conv2 = HeteroConv({
                    ('patient', 'interacts', 'structure'): GATConv((hidden_channels * heads, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                    ('structure', 'rev_interacts', 'patient'): GATConv((hidden_channels * heads, hidden_channels), hidden_channels, heads=heads, add_self_loops=False),
                })

        # Final linear layer for output classification
        self.lin = Linear(hidden_channels * heads, out_channels)

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass through the ClinGRAD model
        
        Args:
            x_dict: Dictionary of node features
            edge_index_dict: Dictionary of edge indices
            
        Returns:
            Tensor: Output predictions for patient nodes
        """
        # Initial node feature transformations
        if 'G' in self.modalities and 'R' in self.modalities:
            x_dict = {
                'patient': self.patient_lin(x_dict['patient']),
                'gene': self.gene_lin(x_dict['gene'].float()),
                'structure': self.structure_lin(x_dict['structure']),
            }
            edge_index_dict[('structure', 'rev_interacts', 'patient')] = edge_index_dict[('patient', 'interacts', 'structure')].flip(0)
            edge_index_dict[('gene', 'rev_associated_with', 'patient')] = edge_index_dict[('patient', 'associated_with', 'gene')].flip(0)

        elif 'G' in self.modalities:
            x_dict = {
                'patient': self.patient_lin(x_dict['patient']),
                'gene': self.gene_lin(x_dict['gene'].float()),
            }
            edge_index_dict[('gene', 'rev_associated_with', 'patient')] = edge_index_dict[('patient', 'associated_with', 'gene')].flip(0)

        elif 'R' in self.modalities:
            x_dict = {
                'patient': self.patient_lin(x_dict['patient']),
                'structure': self.structure_lin(x_dict['structure']),
            }
            edge_index_dict[('structure', 'rev_interacts', 'patient')] = edge_index_dict[('patient', 'interacts', 'structure')].flip(0)

        # Apply the first GAT layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}

        # Apply the second GAT layer
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}

        # Return the output for patient node classification
        return self.lin(x_dict['patient'])