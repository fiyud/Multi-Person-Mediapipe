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