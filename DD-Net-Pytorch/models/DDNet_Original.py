#! /usr/bin/env python
#! coding:utf-8

from utils import poses_motion
import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
import math


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


class DDNet_Original(nn.Module):
    def __init__(self, frame_l, joint_n, joint_d, feat_d, filters, class_num):
        super(DDNet_Original, self).__init__()
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

        # diff_slow part
        self.slow_conv1 = nn.Sequential(
            c1D(frame_l, joint_n * joint_d, 2 * filters, 1),
            spatialDropout1D(0.1)
        )
        self.slow_conv2 = nn.Sequential(
            c1D(frame_l, 2 * filters, filters, 3),
            spatialDropout1D(0.1)
        )
        self.slow_conv3 = c1D(frame_l, filters, filters, 1)
        self.slow_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            spatialDropout1D(0.1)
        )

        # fast_part
        self.fast_conv1 = nn.Sequential(
            c1D(frame_l//2, joint_n * joint_d, 2 * filters, 1), spatialDropout1D(0.1))
        self.fast_conv2 = nn.Sequential(
            c1D(frame_l//2, 2 * filters, filters, 3), spatialDropout1D(0.1))
        self.fast_conv3 = nn.Sequential(
            c1D(frame_l//2, filters, filters, 1), spatialDropout1D(0.1))

        # after cat
        self.block1 = block(frame_l//2, 3 * filters, 2 * filters, 3)
        self.block_pool1 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2), spatialDropout1D(0.1))

        self.block2 = block(frame_l//4, 2 * filters, 4 * filters, 3)
        self.block_pool2 = nn.Sequential(nn.MaxPool1d(
            kernel_size=2), spatialDropout1D(0.1))

        self.block3 = nn.Sequential(
            block(frame_l//8, 4 * filters, 8 * filters, 3), spatialDropout1D(0.1))

        self.linear1 = nn.Sequential(
            d1D(8 * filters, 128),
            nn.Dropout(0.5)
        )
        self.linear2 = nn.Sequential(
            d1D(128, 128),
            nn.Dropout(0.5)
        )

        self.linear3 = nn.Linear(128, class_num)

    def forward(self, M, P=None):
        x = self.jcd_conv1(M)
        x = self.jcd_conv2(x)
        x = self.jcd_conv3(x)
        x = x.permute(0, 2, 1)
        # pool will downsample the D dim of (B,C,D)
        # but we want to downsample the C channels
        # 1x1 conv may be a better choice
        x = self.jcd_pool(x)
        x = x.permute(0, 2, 1)

        diff_slow, diff_fast = poses_motion(P)
        x_d_slow = self.slow_conv1(diff_slow)
        x_d_slow = self.slow_conv2(x_d_slow)
        x_d_slow = self.slow_conv3(x_d_slow)
        x_d_slow = x_d_slow.permute(0, 2, 1)
        x_d_slow = self.slow_pool(x_d_slow)
        x_d_slow = x_d_slow.permute(0, 2, 1)

        x_d_fast = self.fast_conv1(diff_fast)
        x_d_fast = self.fast_conv2(x_d_fast)
        x_d_fast = self.fast_conv3(x_d_fast)
        # x,x_d_fast,x_d_slow shape: (B,framel//2,filters)

        x = torch.cat((x, x_d_slow, x_d_fast), dim=2)
        x = self.block1(x)
        x = x.permute(0, 2, 1)
        x = self.block_pool1(x)
        x = x.permute(0, 2, 1)

        x = self.block2(x)
        x = x.permute(0, 2, 1)
        x = self.block_pool2(x)
        x = x.permute(0, 2, 1)

        x = self.block3(x)
        # max pool over (B,C,D) C channels
        x = torch.max(x, dim=1).values

        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

from torch_geometric.nn import GCNConv

# class GCN1D(nn.Module):
#     # input (B,C,D) //batch,channels,dims
#     # output = (B,C,filters)
#     def __init__(self, input_channels, input_dims, filters, kernel):
#         super(GCN1D, self).__init__()
#         self.gcn = GCNConv(input_dims, filters)
#         self.bn = nn.BatchNorm1d(num_features=input_channels)
        
#     def forward(self, x):
#         batch_size, seq_len, feat_dim = x.shape
        
#         # For GCNConv, we need to provide edge_index
#         # Creating a simple chain graph for sequential data
#         # Each node connects to its neighbors
#         edge_index = []
#         for i in range(seq_len-1):
#             edge_index.append([i, i+1])
#             edge_index.append([i+1, i])  # Bidirectional
            
#         edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#         edge_index = edge_index.to(x.device)
        
#         # Process each batch item separately
#         outputs = []
#         for b in range(batch_size):
#             # Reshape for GCNConv: (seq_len, feat_dim)
#             x_b = x[b]
            
#             # Apply GCNConv
#             out = self.gcn(x_b, edge_index)
#             outputs.append(out)
            
#         # Stack results back to batch
#         output = torch.stack(outputs)
        
#         # Apply normalization and activation
#         output = self.bn(output)
#         output = F.leaky_relu(output, 0.2, True)
        
#         return output


# class block(nn.Module):
#     def __init__(self, input_channels, input_dims, filters, kernel):
#         super(block, self).__init__()
#         self.gcn1 = GCN1D(input_channels, input_dims, filters, kernel)
#         self.gcn2 = GCN1D(input_channels, filters, filters, kernel)

#     def forward(self, x):
#         output = self.gcn1(x)
#         output = self.gcn2(output)
#         return output


# class d1D(nn.Module):
#     def __init__(self, input_dims, filters):
#         super(d1D, self).__init__()
#         self.linear = nn.Linear(input_dims, filters)
#         self.bn = nn.BatchNorm1d(num_features=filters)

#     def forward(self, x):
#         output = self.linear(x)
#         output = self.bn(output)
#         output = F.leaky_relu(output, 0.2)
#         return output


# class spatialDropout1D(nn.Module):
#     def __init__(self, p):
#         super(spatialDropout1D, self).__init__()
#         self.dropout = nn.Dropout2d(p)

#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#         x = self.dropout(x)
#         x = x.permute(0, 2, 1)
#         return x


# class DDNet_GCN(nn.Module):
#     def __init__(self, frame_l, joint_n, joint_d, feat_d, filters, class_num):
#         super(DDNet_GCN, self).__init__()
#         # JCD part
#         self.jcd_conv1 = nn.Sequential(
#             GCN1D(frame_l, feat_d, 2 * filters, 1),
#             spatialDropout1D(0.1)
#         )
#         self.jcd_conv2 = nn.Sequential(
#             GCN1D(frame_l, 2 * filters, filters, 3),
#             spatialDropout1D(0.1)
#         )
#         self.jcd_conv3 = GCN1D(frame_l, filters, filters, 1)
#         self.jcd_pool = nn.Sequential(
#             nn.MaxPool1d(kernel_size=2),
#             spatialDropout1D(0.1)
#         )

#         # diff_slow part
#         self.slow_conv1 = nn.Sequential(
#             GCN1D(frame_l, joint_n * joint_d, 2 * filters, 1),
#             spatialDropout1D(0.1)
#         )
#         self.slow_conv2 = nn.Sequential(
#             GCN1D(frame_l, 2 * filters, filters, 3),
#             spatialDropout1D(0.1)
#         )
#         self.slow_conv3 = GCN1D(frame_l, filters, filters, 1)
#         self.slow_pool = nn.Sequential(
#             nn.MaxPool1d(kernel_size=2),
#             spatialDropout1D(0.1)
#         )

#         # fast_part
#         self.fast_conv1 = nn.Sequential(
#             GCN1D(frame_l//2, joint_n * joint_d, 2 * filters, 1), 
#             spatialDropout1D(0.1)
#         )
#         self.fast_conv2 = nn.Sequential(
#             GCN1D(frame_l//2, 2 * filters, filters, 3), 
#             spatialDropout1D(0.1)
#         )
#         self.fast_conv3 = nn.Sequential(
#             GCN1D(frame_l//2, filters, filters, 1), 
#             spatialDropout1D(0.1)
#         )

#         # after cat
#         self.block1 = block(frame_l//2, 3 * filters, 2 * filters, 3)
#         self.block_pool1 = nn.Sequential(
#             nn.MaxPool1d(kernel_size=2), 
#             spatialDropout1D(0.1)
#         )

#         self.block2 = block(frame_l//4, 2 * filters, 4 * filters, 3)
#         self.block_pool2 = nn.Sequential(
#             nn.MaxPool1d(kernel_size=2), 
#             spatialDropout1D(0.1)
#         )

#         self.block3 = nn.Sequential(
#             block(frame_l//8, 4 * filters, 8 * filters, 3), 
#             spatialDropout1D(0.1)
#         )

#         self.linear1 = nn.Sequential(
#             d1D(8 * filters, 128),
#             nn.Dropout(0.5)
#         )
#         self.linear2 = nn.Sequential(
#             d1D(128, 128),
#             nn.Dropout(0.5)
#         )

#         self.linear3 = nn.Linear(128, class_num)

#     def forward(self, M, P=None):
#         x = self.jcd_conv1(M)
#         x = self.jcd_conv2(x)
#         x = self.jcd_conv3(x)
#         x = x.permute(0, 2, 1)
#         # pool will downsample the D dim of (B,C,D)
#         # but we want to downsample the C channels
#         x = self.jcd_pool(x)
#         x = x.permute(0, 2, 1)

#         diff_slow, diff_fast = poses_motion(P)
#         x_d_slow = self.slow_conv1(diff_slow)
#         x_d_slow = self.slow_conv2(x_d_slow)
#         x_d_slow = self.slow_conv3(x_d_slow)
#         x_d_slow = x_d_slow.permute(0, 2, 1)
#         x_d_slow = self.slow_pool(x_d_slow)
#         x_d_slow = x_d_slow.permute(0, 2, 1)

#         x_d_fast = self.fast_conv1(diff_fast)
#         x_d_fast = self.fast_conv2(x_d_fast)
#         x_d_fast = self.fast_conv3(x_d_fast)
#         # x,x_d_fast,x_d_slow shape: (B,framel//2,filters)

#         x = torch.cat((x, x_d_slow, x_d_fast), dim=2)
#         x = self.block1(x)
#         x = x.permute(0, 2, 1)
#         x = self.block_pool1(x)
#         x = x.permute(0, 2, 1)

#         x = self.block2(x)
#         x = x.permute(0, 2, 1)
#         x = self.block_pool2(x)
#         x = x.permute(0, 2, 1)

#         x = self.block3(x)
#         # max pool over (B,C,D) C channels
#         x = torch.max(x, dim=1).values

#         x = self.linear1(x)
#         x = self.linear2(x)
#         x = self.linear3(x)
#         return x


class DDNet_with_Stats_Stream(nn.Module):
    """
    DDNet architecture with an additional statistical features stream
    with adaptive convolutions and pooling to handle small input sizes.
    """
    def __init__(self, frame_l, joint_n, joint_d, feat_d, filters, num_class):
        super(DDNet_with_Stats_Stream, self).__init__()
        
        # Store dimensions for later use
        self.frame_l = frame_l
        self.joint_n = joint_n
        self.joint_d = joint_d
        
        #================ ORIGINAL DDNET ARCHITECTURE ================
        
        # Stream 1: JCD features (spatial-temporal JCD data)
        self.jcd_stream = nn.Sequential(
            nn.Conv1d(in_channels=frame_l, out_channels=filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(filters),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(filters),
            nn.ReLU(),
        )
        # Use adaptive pooling for JCD stream
        self.pool_jcd_1 = nn.AdaptiveMaxPool1d(feat_d // 3)
        
        self.jcd_stream_2 = nn.Sequential(
            nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(filters),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(filters),
            nn.ReLU(),
        )
        self.pool_jcd_2 = nn.AdaptiveMaxPool1d(feat_d // 6)
        
        # Stream 2: Joint coordinates (pose data)
        self.joint_stream = nn.Sequential(
            nn.Conv2d(in_channels=frame_l, out_channels=filters, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Dropout(0.25),
            # Adjust kernel size based on input dimensions
            nn.Conv2d(in_channels=filters, out_channels=filters, 
                    kernel_size=(min(3, joint_n), min(3, joint_d)), 
                    stride=1, 
                    padding=(min(3, joint_n)//2, min(3, joint_d)//2)),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
        )
        # Adaptive pooling instead of fixed pooling
        self.pool_pose_1 = nn.AdaptiveMaxPool2d((max(5, joint_n // 3), max(2, joint_d // 3)))
        
        # Define a function to create appropriate conv layers based on input size
        def create_adaptive_conv2d(filters, min_size=3):
            kernel_h = min(min_size, max(5, joint_n // 3))
            kernel_w = min(min_size, max(2, joint_d // 3))
            padding_h = kernel_h // 2
            padding_w = kernel_w // 2
            
            return nn.Sequential(
                nn.Conv2d(in_channels=filters, out_channels=filters, 
                        kernel_size=(kernel_h, kernel_w), 
                        stride=1, 
                        padding=(padding_h, padding_w)),
                nn.BatchNorm2d(filters),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Conv2d(in_channels=filters, out_channels=filters, 
                        kernel_size=(kernel_h, kernel_w), 
                        stride=1, 
                        padding=(padding_h, padding_w)),
                nn.BatchNorm2d(filters),
                nn.ReLU(),
            )
        
        # Create adaptive convolution layers
        self.joint_stream_2 = create_adaptive_conv2d(filters)
        
        # Final pooling layer
        self.pool_pose_2 = nn.AdaptiveMaxPool2d((max(1, joint_n // 6), max(1, joint_d // 6)))
        
        # Calculate the flattened feature dimensions after pooling
        jcd_flattened = (feat_d // 6) * filters
        pose_flattened = max(1, joint_n // 6) * max(1, joint_d // 6) * filters
        original_features = jcd_flattened + pose_flattened
        
        #================ NEW STATISTICAL STREAM ================
        
        # Number of statistical features: mean, std, max, min, skewness, kurtosis
        self.num_stats = 6
        
        # Statistical features processing network
        self.stats_stream = nn.Sequential(
            nn.Linear(self.num_stats * joint_n * joint_d, filters * 4),
            nn.BatchNorm1d(filters * 4),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(filters * 4, filters * 2),
            nn.BatchNorm1d(filters * 2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(filters * 2, filters),
            nn.BatchNorm1d(filters),
            nn.ReLU(),
        )
        
        #================ FUSION WITH ADDITIONAL STREAM ================
        
        # Total feature dimensions including the new statistical stream
        total_features = original_features + filters
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(total_features, filters * 2),
            nn.BatchNorm1d(filters * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(filters * 2, num_class)
        )

    def forward(self, M, P):
        batch_size = M.size(0)
        
        # Debug info
        # print(f"M shape: {M.shape}")
        # print(f"P shape: {P.shape}")
        
        #================ PROCESS ORIGINAL STREAMS ================
        
        # Process through JCD stream
        x_jcd = self.jcd_stream(M)
        # print(f"After jcd_stream: {x_jcd.shape}")
        x_jcd = self.pool_jcd_1(x_jcd)
        # print(f"After pool_jcd_1: {x_jcd.shape}")
        x_jcd = self.jcd_stream_2(x_jcd)
        # print(f"After jcd_stream_2: {x_jcd.shape}")
        x_jcd = self.pool_jcd_2(x_jcd)
        # print(f"After pool_jcd_2: {x_jcd.shape}")
        x_jcd = x_jcd.view(batch_size, -1)
        # print(f"Flattened jcd: {x_jcd.shape}")
        
        # Process through Joint stream
        x_joint = self.joint_stream(P)
        # print(f"After joint_stream: {x_joint.shape}")
        x_joint = self.pool_pose_1(x_joint)
        # print(f"After pool_pose_1: {x_joint.shape}")
        x_joint = self.joint_stream_2(x_joint)
        # print(f"After joint_stream_2: {x_joint.shape}")
        x_joint = self.pool_pose_2(x_joint)
        # print(f"After pool_pose_2: {x_joint.shape}")
        x_joint = x_joint.view(batch_size, -1)
        # print(f"Flattened joint: {x_joint.shape}")
        
        # Concatenate original DDNet features
        original_features = torch.cat([x_jcd, x_joint], dim=1)
        # print(f"Original features: {original_features.shape}")
        
        #================ PROCESS NEW STATISTICAL STREAM ================
        
        # Reshape pose data for statistical computation
        # P shape: [batch_size, frame_l, joint_n, joint_d]
        P_reshaped = P.permute(0, 2, 3, 1)  # [batch_size, joint_n, joint_d, frame_l]
        P_flat = P_reshaped.reshape(batch_size, self.joint_n * self.joint_d, self.frame_l)
        
        # Extract statistical features
        # Mean and std
        mean_stats = torch.mean(P_flat, dim=2)  # [batch_size, joint_n * joint_d]
        std_stats = torch.std(P_flat, dim=2)    # [batch_size, joint_n * joint_d]
        
        # Min and max
        min_stats, _ = torch.min(P_flat, dim=2)  # [batch_size, joint_n * joint_d]
        max_stats, _ = torch.max(P_flat, dim=2)  # [batch_size, joint_n * joint_d]
        
        # Skewness and kurtosis (approximate)
        # Center the data
        centered_data = P_flat - mean_stats.unsqueeze(2)
        # Compute moments
        eps = 1e-10  # Small epsilon to avoid division by zero
        var = torch.var(P_flat, dim=2) + eps  # [batch_size, joint_n * joint_d]
        
        # Third moment (for skewness)
        third_moment = torch.mean(centered_data**3, dim=2)  # [batch_size, joint_n * joint_d]
        skewness = third_moment / (torch.sqrt(var)**3 + eps)  # [batch_size, joint_n * joint_d]
        
        # Fourth moment (for kurtosis)
        fourth_moment = torch.mean(centered_data**4, dim=2)  # [batch_size, joint_n * joint_d]
        kurtosis = fourth_moment / (var**2 + eps) - 3.0  # Excess kurtosis (normal = 0)
        
        # Combine all statistical features
        stat_features = torch.cat([
            mean_stats, std_stats, min_stats, max_stats, skewness, kurtosis
        ], dim=1)  # [batch_size, 6 * joint_n * joint_d]
        
        # Process statistical features through its dedicated network
        x_stats = self.stats_stream(stat_features)  # [batch_size, filters]
        # print(f"Statistical features: {x_stats.shape}")
        
        #================ FUSION OF ALL STREAMS ================
        
        # Concatenate all stream features (original DDNet + statistical)
        combined_features = torch.cat([original_features, x_stats], dim=1)
        # print(f"Combined features: {combined_features.shape}")
        
        # Final fusion and classification
        output = self.fusion(combined_features)
        
        return output