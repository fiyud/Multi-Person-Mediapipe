import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DynamicDeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(DynamicDeformableConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.offset_conv = nn.Conv2d(
            in_channels, 2 * kernel_size * kernel_size,
            kernel_size=3, padding=1, bias=True
        )
        
        # Initialize offset prediction to zeros
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        offsets = self.offset_conv(x)  # [B, 2*kh*kw, H, W]
        return self._deformable_conv2d(x, offsets)
    
    def _deformable_conv2d(self, input, offset):
        batch_size, _, in_height, in_width = input.shape
        out_channels = self.out_channels
        
        # output tensor
        output = torch.zeros(
            batch_size, out_channels, 
            (in_height - self.kernel_size + 2 * self.padding) // self.stride + 1,
            (in_width - self.kernel_size + 2 * self.padding) // self.stride + 1,
            device=input.device
        )
        
        # Pad input
        padded_input = F.pad(input, [self.padding] * 4)
        
        # Calculate sampling grid
        grid_h, grid_w = torch.meshgrid(
            torch.arange(self.kernel_size, device=input.device),
            torch.arange(self.kernel_size, device=input.device)
        )
        grid = torch.stack([grid_w, grid_h], dim=-1).float()  # [kh, kw, 2]
        grid = grid.view(-1, 2)  # [kh*kw, 2]
        
        # Grid offset for each position
        h_out = (in_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        w_out = (in_width - self.kernel_size + 2 * self.padding) // self.stride + 1
        
        for b in range(batch_size):
            for h in range(h_out):
                for w in range(w_out):
                    # center position
                    center_h = h * self.stride
                    center_w = w * self.stride
                    
                    # Get offsets for this position
                    offset_h = offset[b, :self.kernel_size * self.kernel_size, h, w]
                    offset_w = offset[b, self.kernel_size * self.kernel_size:, h, w]
                    offset_hw = torch.stack([offset_w, offset_h], dim=-1)  # [kh*kw, 2]
                    
                    # Calculate sampling positions with offsets
                    pos = grid + offset_hw  # [kh*kw, 2]
                    
                    # Adjust to actual positions in padded input
                    pos_w = pos[:, 0] + center_w + self.padding
                    pos_h = pos[:, 1] + center_h + self.padding
                    
                    # Ensure positions are within bounds (0 to padded_size-1)
                    pos_w = torch.clamp(pos_w, 0, padded_input.shape[3] - 1)
                    pos_h = torch.clamp(pos_h, 0, padded_input.shape[2] - 1)
                    
                    # Get integer positions for bilinear interpolation
                    pos_w0 = torch.floor(pos_w).long()
                    pos_h0 = torch.floor(pos_h).long()
                    pos_w1 = torch.clamp(pos_w0 + 1, 0, padded_input.shape[3] - 1)
                    pos_h1 = torch.clamp(pos_h0 + 1, 0, padded_input.shape[2] - 1)
                    
                    # Calculate interpolation weights
                    weight_w0 = pos_w1 - pos_w
                    weight_h0 = pos_h1 - pos_h
                    weight_w1 = pos_w - pos_w0
                    weight_h1 = pos_h - pos_h0
                    
                    # Sample values using bilinear interpolation
                    for c_out in range(out_channels):
                        for c_in in range(self.in_channels):
                            val = torch.zeros(self.kernel_size * self.kernel_size, device=input.device)
                            
                            for i, (p_h0, p_w0, p_h1, p_w1, w_h0, w_w0, w_h1, w_w1) in enumerate(zip(
                                pos_h0, pos_w0, pos_h1, pos_w1, weight_h0, weight_w0, weight_h1, weight_w1
                            )):
                                val[i] = (padded_input[b, c_in, p_h0, p_w0] * w_h0 * w_w0 +
                                         padded_input[b, c_in, p_h0, p_w1] * w_h0 * w_w1 +
                                         padded_input[b, c_in, p_h1, p_w0] * w_h1 * w_w0 +
                                         padded_input[b, c_in, p_h1, p_w1] * w_h1 * w_w1)
                            
                            w_idx = 0
                            for kh in range(self.kernel_size):
                                for kw in range(self.kernel_size):
                                    output[b, c_out, h, w] += val[w_idx] * self.weight[c_out, c_in, kh, kw]
                                    w_idx += 1
        
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        
        return output


class DynamicDeformableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(DynamicDeformableConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.offset_conv = nn.Conv1d(
            in_channels, kernel_size,
            kernel_size=3, padding=1, bias=True
        )
        
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        offsets = self.offset_conv(x)  # [B, k, L]
        return self._deformable_conv1d(x, offsets)
    
    def _deformable_conv1d(self, input, offset):
        batch_size, _, in_length = input.shape
        out_channels = self.out_channels
        
        output = torch.zeros(
            batch_size, out_channels, 
            (in_length - self.kernel_size + 2 * self.padding) // self.stride + 1,
            device=input.device
        )
        
        padded_input = F.pad(input, [self.padding, self.padding])
        grid = torch.arange(self.kernel_size, device=input.device).float()
        
        l_out = (in_length - self.kernel_size + 2 * self.padding) // self.stride + 1
        
        for b in range(batch_size):
            for l in range(l_out):
                center_l = l * self.stride
                
                # Get offsets for this position
                offset_l = offset[b, :, l]  # [k]
                
                # Calculate sampling positions with offsets
                pos = grid + offset_l  # [k]
                
                # Adjust to actual positions in padded input
                pos_l = pos + center_l + self.padding
                
                # Ensure positions are within bounds (0 to padded_size-1)
                pos_l = torch.clamp(pos_l, 0, padded_input.shape[2] - 1)
                
                # Get integer positions for linear interpolation
                pos_l0 = torch.floor(pos_l).long()
                pos_l1 = torch.clamp(pos_l0 + 1, 0, padded_input.shape[2] - 1)
                
                # Calculate interpolation weights
                weight_l0 = pos_l1 - pos_l
                weight_l1 = pos_l - pos_l0
                
                # Sample values using linear interpolation
                for c_out in range(out_channels):
                    for c_in in range(self.in_channels):
                        val = torch.zeros(self.kernel_size, device=input.device)
                        
                        for i, (p_l0, p_l1, w_l0, w_l1) in enumerate(zip(
                            pos_l0, pos_l1, weight_l0, weight_l1
                        )):
                            val[i] = (padded_input[b, c_in, p_l0] * w_l0 +
                                     padded_input[b, c_in, p_l1] * w_l1)
                        
                        for k in range(self.kernel_size):
                            output[b, c_out, l] += val[k] * self.weight[c_out, c_in, k]
        
        if self.bias is not None:
            output += self.bias.view(1, -1, 1)
        
        return output


class DDNetWithDynamicKernels(nn.Module):
    """
    Enhanced DDNet with Dynamic Kernel Deformation.
    This novel approach allows convolution kernels to adapt their shape based on input data,
    significantly improving the model's ability to capture complex motion patterns.
    """
    def __init__(self, frame_l, joint_n, joint_d, feat_d, filters, num_class):
        super(DDNetWithDynamicKernels, self).__init__()
        
        # Store dimensions for later use
        self.frame_l = frame_l
        self.joint_n = joint_n
        self.joint_d = joint_d
        self.feat_d = feat_d
        
        #================ STREAM 1: JCD FEATURES WITH DYNAMIC KERNELS ================
        
        # First conv block with dynamic kernels
        self.jcd_conv1 = DynamicDeformableConv1d(frame_l, filters, kernel_size=3, padding=1)
        self.jcd_bn1 = nn.BatchNorm1d(filters)
        self.jcd_conv2 = DynamicDeformableConv1d(filters, filters, kernel_size=3, padding=1)
        self.jcd_bn2 = nn.BatchNorm1d(filters)
        
        # First pooling
        self.pool_jcd_1 = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)
        
        # Second conv block with dynamic kernels
        self.jcd_conv3 = DynamicDeformableConv1d(filters, filters, kernel_size=3, padding=1)
        self.jcd_bn3 = nn.BatchNorm1d(filters)
        self.jcd_conv4 = DynamicDeformableConv1d(filters, filters, kernel_size=3, padding=1)
        self.jcd_bn4 = nn.BatchNorm1d(filters)
        
        # Second pooling
        self.pool_jcd_2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        #================ STREAM 2: JOINT COORDINATES WITH DYNAMIC KERNELS ================
        
        # First conv block with dynamic kernels
        self.joint_conv1 = nn.Conv2d(frame_l, filters, kernel_size=(1, 1), stride=1, padding=0)
        self.joint_bn1 = nn.BatchNorm2d(filters)
        self.joint_conv2 = DynamicDeformableConv2d(filters, filters, kernel_size=3, padding=1)
        self.joint_bn2 = nn.BatchNorm2d(filters)
        
        # First pooling
        self.pool_pose_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3), padding=(1, 1))
        
        # Second conv block with dynamic kernels
        self.joint_conv3 = DynamicDeformableConv2d(filters, filters, kernel_size=3, padding=1)
        self.joint_bn3 = nn.BatchNorm2d(filters)
        self.joint_conv4 = DynamicDeformableConv2d(filters, filters, kernel_size=3, padding=1)
        self.joint_bn4 = nn.BatchNorm2d(filters)
        
        # Second pooling
        self.pool_pose_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        
        #================ STATISTICAL STREAM ================
        
        # Number of statistical features: mean, std, max, min, skewness, kurtosis
        self.num_stats = 6
        
        # Statistical features processing network
        self.temp_stats_linear1 = nn.Linear(self.num_stats * joint_n * joint_d, filters * 2)
        self.temp_stats_linear2 = nn.Linear(filters * 2, filters)
        self.temp_stats_ln = nn.LayerNorm(filters)
        
        #================ FUSION LAYERS ================
        
        # Calculate the flattened feature dimensions after pooling
        jcd_flattened = (feat_d // 6) * filters
        pose_flattened = ((joint_n // 3) * (joint_d // 3) // 4) * filters
        total_features = jcd_flattened + pose_flattened + filters
        
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
        
        #================ PROCESS JCD STREAM WITH DYNAMIC KERNELS ================
        
        # First conv block
        x_jcd = F.relu(self.jcd_bn1(self.jcd_conv1(M)))
        x_jcd = F.dropout(x_jcd, p=0.25, training=self.training)
        x_jcd = F.relu(self.jcd_bn2(self.jcd_conv2(x_jcd)))
        
        # First pooling
        x_jcd = self.pool_jcd_1(x_jcd)
        
        # Second conv block
        x_jcd = F.relu(self.jcd_bn3(self.jcd_conv3(x_jcd)))
        x_jcd = F.dropout(x_jcd, p=0.25, training=self.training)
        x_jcd = F.relu(self.jcd_bn4(self.jcd_conv4(x_jcd)))
        
        # Second pooling
        x_jcd = self.pool_jcd_2(x_jcd)
        
        # Flatten
        x_jcd = x_jcd.view(batch_size, -1)
        
        #================ PROCESS JOINT STREAM WITH DYNAMIC KERNELS ================
        
        # First conv block
        x_joint = F.relu(self.joint_bn1(self.joint_conv1(P)))
        x_joint = F.dropout(x_joint, p=0.25, training=self.training)
        x_joint = F.relu(self.joint_bn2(self.joint_conv2(x_joint)))
        
        # First pooling
        x_joint = self.pool_pose_1(x_joint)
        
        # Second conv block
        x_joint = F.relu(self.joint_bn3(self.joint_conv3(x_joint)))
        x_joint = F.dropout(x_joint, p=0.25, training=self.training)
        x_joint = F.relu(self.joint_bn4(self.joint_conv4(x_joint)))
        
        # Second pooling
        x_joint = self.pool_pose_2(x_joint)
        
        # Flatten
        x_joint = x_joint.view(batch_size, -1)
        
        #================ PROCESS STATISTICAL STREAM ================
        
        # Reshape pose data for statistical computation
        # P shape: [batch_size, frame_l, joint_n, joint_d]
        x_stats = P.reshape(batch_size, self.frame_l, -1)  # [batch_size, time_step, num_joints * num_features]
        
        # Extract advanced statistical features along time dimension
        # Mean and std
        mean_stats = torch.mean(x_stats, dim=1)  # [batch_size, num_joints * num_features]
        std_stats = torch.std(x_stats, dim=1)    # [batch_size, num_joints * num_features]
        
        # Min and max
        min_stats, _ = torch.min(x_stats, dim=1)  # [batch_size, num_joints * num_features]
        max_stats, _ = torch.max(x_stats, dim=1)  # [batch_size, num_joints * num_features]
        
        # Skewness and kurtosis (approximate)
        # Center the data
        centered_data = x_stats - mean_stats.unsqueeze(1)
        # Compute moments
        eps = 1e-10  # Small epsilon to avoid division by zero
        var = torch.var(x_stats, dim=1) + eps  # [batch_size, num_joints * num_features]
        
        # Third moment (for skewness)
        third_moment = torch.mean(centered_data**3, dim=1)  # [batch_size, num_joints * num_features]
        skewness = third_moment / (torch.sqrt(var)**3 + eps)  # [batch_size, num_joints * num_features]
        
        # Fourth moment (for kurtosis)
        fourth_moment = torch.mean(centered_data**4, dim=1)  # [batch_size, num_joints * num_features]
        kurtosis = fourth_moment / (var**2 + eps) - 3.0  # Excess kurtosis (normal = 0)
        
        # Combine all statistical features
        stat_features = torch.cat([
            mean_stats, std_stats, min_stats, max_stats, skewness, kurtosis
        ], dim=1)  # [batch_size, 6 * num_joints * num_features]
        
        # Process with MLP
        stat_features = F.relu(self.temp_stats_linear1(stat_features))
        stat_features = self.temp_stats_linear2(stat_features)  # [batch_size, hidden_dim]
        stat_features = self.temp_stats_ln(stat_features)
        
        #================ FUSION OF ALL STREAMS ================
        
        # Concatenate all features
        combined_features = torch.cat([x_jcd, x_joint, stat_features], dim=1)
        
        # Final fusion and classification
        output = self.fusion(combined_features)
        
        return output


# Example usage
if __name__ == "__main__":
    # Example dimensions
    frame_l = 32  # Number of frames
    joint_n = 15  # Number of joints
    joint_d = 2   # Joint dimensions (2D or 3D)
    feat_d = 105  # JCD feature dimension
    filters = 64  # Number of filters
    num_class = 21  # Number of action classes
    
    # Create the model
    model = DDNetWithDynamicKernels(frame_l, joint_n, joint_d, feat_d, filters, num_class)
    
    # Generate random input tensors for testing
    batch_size = 4
    M = torch.randn(batch_size, frame_l, feat_d)  # JCD features
    P = torch.randn(batch_size, frame_l, joint_n, joint_d)  # Joint coordinates
    
    # Forward pass
    output = model(M, P)
    print(f"Input shapes: M={M.shape}, P={P.shape}")
    print(f"Output shape: {output.shape}")