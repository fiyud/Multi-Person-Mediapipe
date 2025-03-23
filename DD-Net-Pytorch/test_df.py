import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np

# First, let's copy the DynamicDeformableConv2d implementation
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


# Now let's create a function to visualize a feature map
def visualize_feature_map(feature_map, title, subplot_idx=None, cmap='viridis'):
    if subplot_idx is not None:
        plt.subplot(subplot_idx)
    
    # Take the first batch and channel if they exist
    if len(feature_map.shape) == 4:
        feature_map = feature_map[0, 0]  # Take first batch, first channel
    elif len(feature_map.shape) == 3:
        feature_map = feature_map[0]  # Take first channel
    
    plt.imshow(feature_map.detach().cpu().numpy(), cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.axis('on')


# Function to visualize offsets
def visualize_offsets(offsets, h_idx, w_idx, kernel_size=3, subplot_idx=None):
    if subplot_idx is not None:
        plt.subplot(subplot_idx)
    
    # Extract offsets for a specific position (h_idx, w_idx)
    offset_h = offsets[0, :kernel_size*kernel_size, h_idx, w_idx].detach().cpu().numpy()
    offset_w = offsets[0, kernel_size*kernel_size:, h_idx, w_idx].detach().cpu().numpy()
    
    # Create grid points
    grid_h, grid_w = np.meshgrid(
        np.arange(kernel_size),
        np.arange(kernel_size)
    )
    grid_h = grid_h.flatten()
    grid_w = grid_w.flatten()
    
    # Calculate deformed positions
    pos_h = grid_h + offset_h
    pos_w = grid_w + offset_w
    
    # Draw the regular grid
    plt.scatter(grid_w, grid_h, c='blue', marker='o', s=100, alpha=0.5, label='Regular Grid')
    
    # Draw the deformed grid
    plt.scatter(pos_w, pos_h, c='red', marker='x', s=100, label='Deformed Grid')
    
    # Draw arrows from regular to deformed positions
    for i in range(len(grid_h)):
        plt.arrow(grid_w[i], grid_h[i], pos_w[i]-grid_w[i], pos_h[i]-grid_h[i], 
                 head_width=0.1, head_length=0.1, fc='black', ec='black', alpha=0.5)
    
    plt.xlim(-0.5, kernel_size-0.5)
    plt.ylim(-0.5, kernel_size-0.5)
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.legend()
    plt.title(f'Offset Field at Position ({h_idx}, {w_idx})')
    plt.grid(True)


# Create a dummy input - a simple 2-channel feature map
def create_dummy_input(height=8, width=8, pattern_type='gradient'):
    # Create a batch with 1 sample, 2 channels
    dummy_input = torch.zeros(1, 2, height, width)
    
    if pattern_type == 'gradient':
        # Channel 0: Horizontal gradient
        for h in range(height):
            for w in range(width):
                dummy_input[0, 0, h, w] = w / (width - 1)
        
        # Channel 1: Vertical gradient
        for h in range(height):
            for w in range(width):
                dummy_input[0, 1, h, w] = h / (height - 1)
    
    elif pattern_type == 'checkerboard':
        # Create a checkerboard pattern
        for h in range(height):
            for w in range(width):
                if (h + w) % 2 == 0:
                    dummy_input[0, 0, h, w] = 1.0
                    dummy_input[0, 1, h, w] = 0.0
                else:
                    dummy_input[0, 0, h, w] = 0.0
                    dummy_input[0, 1, h, w] = 1.0
    
    elif pattern_type == 'circle':
        # Create a circular pattern
        center_h, center_w = height // 2, width // 2
        max_radius = min(height, width) // 2
        
        for h in range(height):
            for w in range(width):
                # Distance from center
                dist = math.sqrt((h - center_h)**2 + (w - center_w)**2)
                # Normalize distance
                norm_dist = dist / max_radius
                
                # Channel 0: Circle (higher in center)
                dummy_input[0, 0, h, w] = max(0, 1 - norm_dist)
                
                # Channel 1: Inverse circle (higher at edges)
                dummy_input[0, 1, h, w] = min(1, norm_dist)
    
    return dummy_input


# Function to demonstrate modified offsets
def create_modified_offsets(model, input_tensor, offset_scale=1.0, pattern='radial'):
    """
    Create modified offsets to demonstrate the effect of deformation.
    
    Args:
        model: DynamicDeformableConv2d model
        input_tensor: Input tensor
        offset_scale: Scale factor for the offset values
        pattern: Pattern for the offsets ('radial', 'expand', 'rotate')
    
    Returns:
        Modified offset tensor
    """
    # First get the original offsets from the model
    with torch.no_grad():
        original_offsets = model.offset_conv(input_tensor)
    
    # Create new offsets based on the pattern
    b, c, h, w = original_offsets.shape
    kernel_size = model.kernel_size
    offsets = torch.zeros_like(original_offsets)
    
    if pattern == 'radial':
        # Radial pattern - points move outward from center of each kernel
        for hi in range(h):
            for wi in range(w):
                for ki in range(kernel_size * kernel_size):
                    # Calculate kernel position (kh, kw) from linear index ki
                    kh = ki // kernel_size
                    kw = ki % kernel_size
                    
                    # Distance from kernel center
                    center_h = (kernel_size - 1) / 2
                    center_w = (kernel_size - 1) / 2
                    dy = kh - center_h
                    dx = kw - center_w
                    
                    # Set offset to move outward
                    offsets[0, ki, hi, wi] = dy * offset_scale  # y-offset
                    offsets[0, ki + kernel_size*kernel_size, hi, wi] = dx * offset_scale  # x-offset
    
    elif pattern == 'expand':
        # Expansion pattern - all points move away from center of each kernel
        for hi in range(h):
            for wi in range(w):
                for ki in range(kernel_size * kernel_size):
                    # Calculate kernel position (kh, kw) from linear index ki
                    kh = ki // kernel_size
                    kw = ki % kernel_size
                    
                    # Distance from kernel center
                    center_h = (kernel_size - 1) / 2
                    center_w = (kernel_size - 1) / 2
                    dy = kh - center_h
                    dx = kw - center_w
                    
                    # Calculate distance from center
                    dist = math.sqrt(dy**2 + dx**2)
                    if dist > 0:
                        # Normalize and scale
                        scale = offset_scale * dist
                        offsets[0, ki, hi, wi] = dy * scale / dist  # y-offset
                        offsets[0, ki + kernel_size*kernel_size, hi, wi] = dx * scale / dist  # x-offset
    
    elif pattern == 'rotate':
        # Rotation pattern - rotate points around kernel center
        angle = math.pi / 4  # 45 degrees
        for hi in range(h):
            for wi in range(w):
                for ki in range(kernel_size * kernel_size):
                    # Calculate kernel position (kh, kw) from linear index ki
                    kh = ki // kernel_size
                    kw = ki % kernel_size
                    
                    # Distance from kernel center
                    center_h = (kernel_size - 1) / 2
                    center_w = (kernel_size - 1) / 2
                    dy = kh - center_h
                    dx = kw - center_w
                    
                    # Rotate by angle
                    new_dx = dx * math.cos(angle) - dy * math.sin(angle)
                    new_dy = dx * math.sin(angle) + dy * math.cos(angle)
                    
                    # Set offset to achieve rotation
                    offsets[0, ki, hi, wi] = (new_dy - dy) * offset_scale  # y-offset
                    offsets[0, ki + kernel_size*kernel_size, hi, wi] = (new_dx - dx) * offset_scale  # x-offset
    
    return offsets


# Main function to demonstrate the deformable convolution
def demonstrate_deformable_conv2d():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create a dummy input tensor
    input_tensor = create_dummy_input(height=8, width=8, pattern_type='circle')
    
    # Initialize the deformable convolution layer
    deform_conv = DynamicDeformableConv2d(
        in_channels=2,
        out_channels=1,
        kernel_size=3,
        stride=1,
        padding=1
    )
    
    # Forward pass with regular offsets (these will be close to zero due to initialization)
    output_normal = deform_conv(input_tensor)
    
    # Get the generated offsets
    offsets_normal = deform_conv.offset_conv(input_tensor)
    
    # Create custom offsets with different patterns
    offsets_radial = create_modified_offsets(deform_conv, input_tensor, offset_scale=0.5, pattern='radial')
    offsets_expand = create_modified_offsets(deform_conv, input_tensor, offset_scale=0.5, pattern='expand')
    offsets_rotate = create_modified_offsets(deform_conv, input_tensor, offset_scale=0.5, pattern='rotate')
    
    # Forward pass with custom offsets
    output_radial = deform_conv._deformable_conv2d(input_tensor, offsets_radial)
    output_expand = deform_conv._deformable_conv2d(input_tensor, offsets_expand)
    output_rotate = deform_conv._deformable_conv2d(input_tensor, offsets_rotate)
    
    # Visualize the results
    plt.figure(figsize=(20, 16))
    
    # Input channels
    visualize_feature_map(input_tensor[:, 0:1], 'Input Channel 0 (Circle Center)', 241)
    visualize_feature_map(input_tensor[:, 1:2], 'Input Channel 1 (Circle Edge)', 242)
    
    # Normal output and offset visualization
    visualize_feature_map(output_normal, 'Output with Normal Offsets', 243)
    visualize_offsets(offsets_normal, 4, 4, kernel_size=3, subplot_idx=244)
    
    # Radial pattern
    visualize_feature_map(output_radial, 'Output with Radial Offsets', 245)
    visualize_offsets(offsets_radial, 4, 4, kernel_size=3, subplot_idx=246)
    
    # Expansion pattern
    visualize_feature_map(output_expand, 'Output with Expansion Offsets', 247)
    visualize_offsets(offsets_expand, 4, 4, kernel_size=3, subplot_idx=248)
    
    plt.tight_layout()
    plt.show()
    
    # Print shapes for verification
    print(f"Input shape: {input_tensor.shape}")
    print(f"Offsets shape: {offsets_normal.shape}")
    print(f"Output shape: {output_normal.shape}")
    
    # Return the results for further analysis if needed
    return {
        'input': input_tensor,
        'output_normal': output_normal,
        'output_radial': output_radial,
        'output_expand': output_expand,
        'output_rotate': output_rotate,
        'offsets_normal': offsets_normal,
        'offsets_radial': offsets_radial,
        'offsets_expand': offsets_expand,
        'offsets_rotate': offsets_rotate,
    }

# Run the demonstration
if __name__ == "__main__":
    results = demonstrate_deformable_conv2d()