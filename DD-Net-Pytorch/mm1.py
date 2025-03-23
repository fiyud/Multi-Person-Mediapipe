from utils import poses_motion
import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
import math
from torch_geometric.nn import GCNConv

from utils import poses_motion
import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
import math
from torch_geometric.nn import GCNConv


class SkeletonGraphConv(nn.Module):
    """
    Graph Convolutional Network module for skeleton data.
    This module applies GCNConv to the skeleton joints at each frame.
    """
    def __init__(self, in_channels, out_channels):
        super(SkeletonGraphConv, self).__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x, edge_index, batch_size, num_frames, num_joints):
        """
        x: Input features of shape (batch_size * num_frames * num_joints, in_channels)
        edge_index: Graph connectivity of shape (2, num_edges)
        """
        # Apply GCN to each frame
        x = self.gcn(x, edge_index)
        x = self.bn(x)
        x = F.leaky_relu(x, 0.2)
        
        # Reshape back to (batch_size, num_frames, num_joints, out_channels)
        x = x.view(batch_size, num_frames, num_joints, -1)
        
        return x


def create_skeleton_graph(num_joints, skeleton_edges=None):
    """
    Create a skeleton graph edge index tensor.
    
    Args:
        num_joints: Number of joints in the skeleton
        skeleton_edges: List of tuples (source, target) representing skeleton connections.
                        If None, a default skeleton structure will be created.
    
    Returns:
        edge_index: A tensor of shape (2, num_edges) containing the edge indices
    """
    if skeleton_edges is None:
        # Define a default skeleton structure if not provided
        # This is just an example - you should adjust this based on your actual skeleton structure
        skeleton_edges = [
            # Torso
            (0, 1), (1, 2), (2, 3),  # Spine
            # Arms
            (1, 4), (4, 5), (5, 6),  # Left arm
            (1, 7), (7, 8), (8, 9),  # Right arm
            # Legs
            (0, 10), (10, 11), (11, 12),  # Left leg
            (0, 13), (13, 14), (14, 15),  # Right leg
        ]
        
        # Add reverse edges to make the graph undirected
        skeleton_edges += [(j, i) for (i, j) in skeleton_edges]
    
    # Convert to tensor
    edge_index = torch.tensor(skeleton_edges, dtype=torch.long).t().contiguous()
    
    return edge_index


class GCNGRUBlock(nn.Module):
    """
    Block that combines GCN processing with GRU for temporal modeling
    """
    def __init__(self, joint_n, joint_d, filters, hidden_dim, dropout_rate=0.1):
        super(GCNGRUBlock, self).__init__()
        self.joint_n = joint_n
        self.joint_d = joint_d
        self.hidden_dim = hidden_dim
        
        # GCN layers
        self.joint_gcn1 = SkeletonGraphConv(joint_d, filters)
        self.joint_gcn2 = SkeletonGraphConv(filters, filters)
        
        # GRU layer for temporal modeling
        self.gru = nn.GRU(
            input_size=joint_n * filters,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0 if dropout_rate == 0 else dropout_rate
        )
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, filters),  # *2 for bidirectional
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x, edge_index, batch_size, frame_l):
        """
        x: Input tensor of shape (batch_size, frame_l, joint_n, joint_d)
        edge_index: Graph connectivity
        """
        # Reshape for GCN: (batch_size * frame_l * joint_n, joint_d)
        x_reshaped = x.reshape(-1, self.joint_d)
        
        # Apply GCN layers
        x_gcn = self.joint_gcn1(x_reshaped, edge_index, batch_size, frame_l, self.joint_n)
        x_gcn = self.joint_gcn2(x_gcn.reshape(-1, x_gcn.shape[-1]), edge_index, batch_size, frame_l, self.joint_n)
        
        # Reshape for GRU: (batch_size, frame_l, joint_n * filters)
        x_gru_in = x_gcn.reshape(batch_size, frame_l, self.joint_n * x_gcn.shape[-1])
        
        # Apply GRU
        x_gru_out, _ = self.gru(x_gru_in)  # (batch_size, frame_l, hidden_dim*2)
        
        # Apply projection to each timestamp
        # First reshape to (batch_size * frame_l, hidden_dim*2)
        x_proj_in = x_gru_out.reshape(-1, x_gru_out.shape[-1])
        x_proj_out = self.projection(x_proj_in)  # (batch_size * frame_l, filters)
        
        # Reshape back to (batch_size, frame_l, filters)
        x_out = x_proj_out.reshape(batch_size, frame_l, -1)
        
        return x_out


class c1D(nn.Module):
    # input (B,C,D) //batch,channels,dims
    # output = (B,C,filters)
    def __init__(self, input_channels, input_dims, filters, kernel):
        super(c1D, self).__init__()
        self.cut_last_element = (kernel % 2 == 0)
        self.padding = math.ceil((kernel - 1)/2)
        self.conv1 = nn.Conv1d(input_dims, filters,
                               kernel, bias=False, padding=self.padding)
        self.bn = nn.BatchNorm1d(num_features=input_channels)

    def forward(self, x):
        # x (B,D,C)
        x = x.permute(0, 2, 1)
        # output (B,filters,C)
        if(self.cut_last_element):
            output = self.conv1(x)[:, :, :-1]
        else:
            output = self.conv1(x)
        # output = (B,C,filters)
        output = output.permute(0, 2, 1)
        output = self.bn(output)
        output = F.leaky_relu(output, 0.2, True)
        return output


class block(nn.Module):
    def __init__(self, input_channels, input_dims, filters, kernel):
        super(block, self).__init__()
        self.c1D1 = c1D(input_channels, input_dims, filters, kernel)
        self.c1D2 = c1D(input_channels, filters, filters, kernel)

    def forward(self, x):
        output = self.c1D1(x)
        output = self.c1D2(output)
        return output


class d1D(nn.Module):
    def __init__(self, input_dims, filters):
        super(d1D, self).__init__()
        self.linear = nn.Linear(input_dims, filters)
        self.bn = nn.BatchNorm1d(num_features=filters)

    def forward(self, x):
        output = self.linear(x)
        output = self.bn(output)
        output = F.leaky_relu(output, 0.2)
        return output


class spatialDropout1D(nn.Module):
    def __init__(self, p):
        super(spatialDropout1D, self).__init__()
        self.dropout = nn.Dropout2d(p)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x


class DDNet_GCN_GRU(nn.Module):
    def __init__(self, frame_l, joint_n, joint_d, feat_d, filters, class_num, gru_hidden_dim=128, skeleton_edges=None):
        super(DDNet_GCN_GRU, self).__init__()
        
        # Create skeleton graph
        self.edge_index = create_skeleton_graph(joint_n, skeleton_edges)
        self.joint_n = joint_n
        self.joint_d = joint_d
        self.frame_l = frame_l
        
        # JCD part
        self.jcd_conv1 = nn.Sequential(
            c1D(frame_l, feat_d, 2 * filters, 1),
            spatialDropout1D(0.1)
        )
        self.jcd_conv2 = nn.Sequential(
            c1D(frame_l, 2 * filters, filters, 3),
            spatialDropout1D(0.1)
        )
        self.jcd_conv3 = c1D(frame_l, filters, filters, 1)
        self.jcd_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            spatialDropout1D(0.1)
        )

        # GCN-GRU-based slow stream
        self.gcn_gru_slow = GCNGRUBlock(
            joint_n=joint_n, 
            joint_d=joint_d, 
            filters=filters,
            hidden_dim=gru_hidden_dim, 
            dropout_rate=0.1
        )
        self.slow_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            spatialDropout1D(0.1)
        )
        
        # GCN-GRU-based fast stream
        self.gcn_gru_fast = GCNGRUBlock(
            joint_n=joint_n, 
            joint_d=joint_d, 
            filters=filters,
            hidden_dim=gru_hidden_dim, 
            dropout_rate=0.1
        )
        
        # Convolutional blocks after concatenation
        self.block1 = block(frame_l//2, 3 * filters, 2 * filters, 3)
        self.block_pool1 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2), 
            spatialDropout1D(0.1)
        )

        self.block2 = block(frame_l//4, 2 * filters, 4 * filters, 3)
        self.block_pool2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2), 
            spatialDropout1D(0.1)
        )

        # GRU after convolutional blocks
        self.final_gru = nn.GRU(
            input_size=4 * filters,
            hidden_size=4 * filters,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Classification head
        self.linear1 = nn.Sequential(
            d1D(4 * filters, 128),
            nn.Dropout(0.5)
        )
        self.linear2 = nn.Sequential(
            d1D(128, 128),
            nn.Dropout(0.5)
        )
        self.linear3 = nn.Linear(128, class_num)

    def forward(self, M, P):
        batch_size = P.shape[0]
        
        # Process JCD features
        x_jcd = self.jcd_conv1(M)
        x_jcd = self.jcd_conv2(x_jcd)
        x_jcd = self.jcd_conv3(x_jcd)
        x_jcd = x_jcd.permute(0, 2, 1)
        x_jcd = self.jcd_pool(x_jcd)
        x_jcd = x_jcd.permute(0, 2, 1)  # (batch_size, frame_l//2, filters)

        # Get edge index for GCN
        edge_index_batch = self.edge_index.to(P.device)
        
        # Process slow stream with GCN-GRU
        x_slow_gcn_gru = self.gcn_gru_slow(P, edge_index_batch, batch_size, self.frame_l)
        x_slow_gcn_gru = x_slow_gcn_gru.permute(0, 2, 1)
        x_slow_gcn_gru = self.slow_pool(x_slow_gcn_gru)  # Downsample to match JCD
        x_slow_gcn_gru = x_slow_gcn_gru.permute(0, 2, 1)  # (batch_size, frame_l//2, filters)
        
        # Process fast stream with GCN-GRU (every other frame)
        P_fast = P[:, ::2, :, :]  # Subsample frames
        x_fast_gcn_gru = self.gcn_gru_fast(P_fast, edge_index_batch, batch_size, self.frame_l//2)
        # Already at (batch_size, frame_l//2, filters)
        
        # Concatenate all features (JCD, GCN-GRU slow, GCN-GRU fast)
        x_cat = torch.cat((x_jcd, x_slow_gcn_gru, x_fast_gcn_gru), dim=2)  # (batch_size, frame_l//2, 3*filters)
        
        # Process with convolutional blocks
        x = self.block1(x_cat)
        x = x.permute(0, 2, 1)
        x = self.block_pool1(x)
        x = x.permute(0, 2, 1)  # (batch_size, frame_l//4, 2*filters)

        x = self.block2(x)
        x = x.permute(0, 2, 1)
        x = self.block_pool2(x)
        x = x.permute(0, 2, 1)  # (batch_size, frame_l//8, 4*filters)
        
        # Process with final GRU
        _, h_n = self.final_gru(x)  # Use the final hidden state
        # h_n shape: (num_layers, batch_size, hidden_size)
        # Take the last layer's hidden state
        x_gru = h_n[-1]  # (batch_size, 4*filters)
        
        # Classification layers
        x = self.linear1(x_gru)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

def poses_diff(x):
    _, H, W, _ = x.shape

    # x.shape (batch,channel,joint_num,joint_dim)
    x = x[:, 1:, ...] - x[:, :-1, ...]

    # x.shape (batch,joint_dim,channel,joint_num,)
    x = x.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(H, W),
                      align_corners=False, mode='bilinear')
    x = x.permute(0, 2, 3, 1)
    # x.shape (batch,channel,joint_num,joint_dim)
    return x


def poses_motion(P):
    """
    Calculate slow and fast motion features.
    
    Args:
        P: Input pose sequence of shape (batch_size, num_frames, num_joints, joint_dims)
    
    Returns:
        P_diff_slow: Slow motion features of shape (batch_size, num_frames-1, joint_n * joint_d)
        P_diff_fast: Fast motion features of shape (batch_size, num_frames//2-1, joint_n * joint_d)
    """
    P_diff_slow = poses_diff(P)
    P_diff_slow = torch.flatten(P_diff_slow, start_dim=2)
    
    P_fast = P[:, ::2, :, :]  # Subsample frames (take every other frame)
    P_diff_fast = poses_diff(P_fast)
    P_diff_fast = torch.flatten(P_diff_fast, start_dim=2)
    
    return P_diff_slow, P_diff_fast