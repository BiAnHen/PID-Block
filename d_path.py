class PIDGate(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        # 确保通道数至少为1
        squeezed_c = max(1, c // r)
        self.fc1 = nn.Conv2d(c, squeezed_c, 1, bias=False)
        self.fc2 = nn.Conv2d(squeezed_c, c, 1, bias=False)
    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        g = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return g

# ===============================
# D 分支的六种实现
# ===============================

class D_SobelLap(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        # 适配层处理 stride 和通道变化
        self.adapter = nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False) if in_c != out_c or stride != 1 else (nn.AvgPool2d(stride) if stride !=1 else nn.Identity())
        
        sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
        sobel_y = sobel_x.t()
        lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32)
        k = torch.stack([sobel_x, sobel_y, lap])[:,None,:,:]
        self.register_buffer("k", k)
        
        self.dw = nn.Conv2d(3*out_c, 3*out_c, 3, padding=1, groups=3*out_c, bias=False)
        self.pw = nn.Conv2d(3*out_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x_adapted = self.adapter(x)
        B, C, H, W = x_adapted.shape
        
        x_rep = x_adapted.repeat_interleave(3, dim=1) # (B, 3*C, H, W)
        k_rep = self.k.repeat(C, 1, 1, 1) # (3*C, 1, 3, 3)
        
        d = F.conv2d(x_rep, k_rep, padding=1, groups=3*C)
        d = self.pw(self.dw(d))
        return self.bn(d)

class D_Bandpass(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.adapter = nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False) if in_c != out_c or stride != 1 else nn.Identity()
        self.conv = nn.Conv2d(out_c, out_c, 3, padding=1, groups=out_c, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x_adapted = self.adapter(x)
        low = F.avg_pool2d(x_adapted, 3, stride=1, padding=1)
        hp = x_adapted - low
        hp = self.conv(hp)
        return self.bn(hp)

class D_DynamicHPF(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.adapter = nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False) if in_c != out_c or stride != 1 else nn.Identity()
        self.base = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.gate = nn.Conv2d(out_c, out_c, 1)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x_adapted = self.adapter(x)
        g = torch.sigmoid(self.gate(x_adapted))
        d = self.base(x_adapted)
        return self.bn(d * g)

class D_Steerable(nn.Module):
    def __init__(self, in_c, out_c, stride=1, num_directions=4):
        super().__init__()
        self.adapter = nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False) if in_c != out_c or stride != 1 else nn.Identity()
        self.dir_convs = nn.ModuleList([
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False) for _ in range(num_directions)
        ])
        self.attn = nn.Conv2d(out_c, num_directions, 1)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x_adapted = self.adapter(x)
        attn = torch.softmax(self.attn(x_adapted), dim=1)
        outs = [conv(x_adapted) for conv in self.dir_convs]
        out = sum(o * attn[:,i:i+1] for i,o in enumerate(outs))
        return self.bn(out)

class D_AntiAlias(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.adapter = nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False) if in_c != out_c or stride != 1 else nn.Identity()
        self.hpf = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        k = torch.tensor([[1,2,1],[2,4,2],[1,2,1]], dtype=torch.float32)/16
        self.register_buffer("blur", k[None,None,:,:])
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x_adapted = self.adapter(x)
        C = x_adapted.size(1)
        x_blur = F.conv2d(x_adapted, self.blur.repeat(C,1,1,1), padding=1, groups=C)
        d = self.hpf(x_blur)
        return self.bn(d)

class D_FADC(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.adapter = nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False) if in_c != out_c or stride != 1 else nn.Identity()
        self.low = nn.Conv2d(out_c, out_c, 3, padding=2, dilation=2, bias=False)
        self.high = nn.Conv2d(out_c, out_c, 3, padding=1, dilation=1, bias=False)
        self.attn = nn.Conv2d(out_c, 2, 1)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x_adapted = self.adapter(x)
        outs = [self.low(x_adapted), self.high(x_adapted)]
        w = torch.softmax(self.attn(x_adapted), dim=1)
        out = outs[0] * w[:,0:1] + outs[1] * w[:,1:2]
        return self.bn(out)
    
class D_MorphGrad(nn.Module):
    def __init__(self, in_c, out_c, stride=1, k=3):
        super().__init__()
        self.adapter = nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False) if (in_c!=out_c or stride!=1) else nn.Identity()
        self.pad = k // 2
        self.dilate = nn.MaxPool2d(kernel_size=k, stride=1, padding=self.pad)
        self.erode = nn.MaxPool2d(kernel_size=k, stride=1, padding=self.pad)
        self.dw = nn.Conv2d(out_c, out_c, 3, padding=1, groups=out_c, bias=False)
        self.pw = nn.Conv2d(out_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
    def forward(self, x):
        x = self.adapter(x)
        dil = self.dilate(x)
        ero = -self.erode(-x)
        mg = dil - ero
        y = self.pw(self.dw(mg))
        return self.bn(y)

class D_WaveletHP(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.adapter = nn.Conv2d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()
        
        # Haar filters (for stride=2 downsampling)
        lh = torch.tensor([[0.5, 0.5],[-0.5,-0.5]], dtype=torch.float32)
        hl = torch.tensor([[0.5,-0.5],[0.5,-0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5,-0.5],[-0.5,0.5]], dtype=torch.float32)
        self.register_buffer("lh", lh[None,None,:,:])
        self.register_buffer("hl", hl[None,None,:,:])
        self.register_buffer("hh", hh[None,None,:,:])
        
        self.proj = nn.Conv2d(3*out_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.stride = stride

    def forward(self, x):
        x = self.adapter(x)
        if self.stride == 1:
            low = F.avg_pool2d(x, 3, stride=1, padding=1)
            return self.bn(x - low)

        C = x.size(1)
        def filt(k): return F.conv2d(x, k.repeat(C,1,1,1), stride=2, groups=C)
        
        LH, HL, HH = filt(self.lh), filt(self.hl), filt(self.hh)
        hp = torch.cat([LH, HL, HH], dim=1)
        
        return self.bn(self.proj(hp))

class D_BoundaryAttn(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.adapter = nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False) if (in_c!=out_c or stride!=1) else nn.Identity()
        self.edge = nn.Sequential(nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_c, 1, 1))
        self.hpf  = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
    def forward(self, x):
        x = self.adapter(x)
        e = torch.sigmoid(self.edge(x))
        d = self.hpf(x)
        return self.bn(d * e)

class PIDBlock_SobelLap(PIDBlock_Base):
    def __init__(self, in_planes, planes, stride=1): super().__init__(in_planes, planes, stride, D_SobelLap)
class PIDBlock_Bandpass(PIDBlock_Base):
    def __init__(self, in_planes, planes, stride=1): super().__init__(in_planes, planes, stride, D_Bandpass)
class PIDBlock_DynamicHPF(PIDBlock_Base):
    def __init__(self, in_planes, planes, stride=1): super().__init__(in_planes, planes, stride, D_DynamicHPF)
class PIDBlock_Steerable(PIDBlock_Base):
    def __init__(self, in_planes, planes, stride=1): super().__init__(in_planes, planes, stride, D_Steerable)
class PIDBlock_AntiAlias(PIDBlock_Base):
    def __init__(self, in_planes, planes, stride=1): super().__init__(in_planes, planes, stride, D_AntiAlias)
class PIDBlock_FADC(PIDBlock_Base):
    def __init__(self, in_planes, planes, stride=1): super().__init__(in_planes, planes, stride, D_FADC)
class PIDBlock_MorphGrad(PIDBlock_Base):
    def __init__(self, in_planes, planes, stride=1): super().__init__(in_planes, planes, stride, D_MorphGrad)
class PIDBlock_WaveletHP(PIDBlock_Base):
    def __init__(self, in_planes, planes, stride=1): super().__init__(in_planes, planes, stride, D_WaveletHP)
class PIDBlock_BoundaryAttn(PIDBlock_Base):
    def __init__(self, in_planes, planes, stride=1): super().__init__(in_planes, planes, stride, D_BoundaryAttn)


class MyBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(MyBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
        
class GenericNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, base_planes=64):
        super(GenericNet, self).__init__()
        self.in_planes = base_planes
        self.conv1 = nn.Conv2d(3, base_planes, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, base_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_planes * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_planes * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, base_planes * 8, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(base_planes * 8 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out); out = self.layer2(out)
        out = self.layer3(out); out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out
