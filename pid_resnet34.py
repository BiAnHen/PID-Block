import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import math
import random
import numpy as np
import argparse
import csv
import os

# --- 0. 随机种子和设备设置 ---
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# --- CSV Logger ---
class CSVLogger:
    def __init__(self, filepath, filename, headers):
        self.filepath = filepath; self.filename = filename
        os.makedirs(self.filepath, exist_ok=True)
        with open(os.path.join(self.filepath, self.filename), 'w', newline='') as f:
            csv.writer(f).writerow(headers)
    def log(self, data_row):
        with open(os.path.join(self.filepath, self.filename), 'a', newline='') as f:
            csv.writer(f).writerow(data_row)

# ==================== SE 模块 ====================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
# ==========================================================

# ==================== ECA 模块 ====================
class ECABlock(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECABlock, self).__init__()
        # 根据通道数自适应计算1D卷积核大小
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化
        y = self.avg_pool(x)
        # Reshape for 1D conv: (B, C, 1, 1) -> (B, 1, C)
        y = y.squeeze(-1).transpose(-1, -2)
        # 1D 卷积
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        # Sigmoid 激活
        y = self.sigmoid(y)
        # 将注意力权重应用到原始特征图
        return x * y.expand_as(x)
# ==========================================================

class MyBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(MyBasicBlock, self).__init__()
        self.residual_path = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))
    def forward(self, x):
        identity = self.shortcut(x)
        residual_out = self.residual_path(x)
        out = F.relu(identity + residual_out)
        return out

# ====================  SE-ResNet 基础块 ====================
class SEBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(SEBasicBlock, self).__init__()
        self.residual_path = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.se = SEBlock(planes) # 在残差路径后加入SE模块
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        residual_out = self.residual_path(x)
        se_out = self.se(residual_out) # 对残差输出应用SE
        out = F.relu(identity + se_out) # SE模块输出与恒等映射相加
        return out
# ==============================================================

# ====================  ECA-ResNet 基础块 ====================
class ECABasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(ECABasicBlock, self).__init__()
        self.residual_path = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.eca = ECABlock(planes) # 在残差路径后加入ECA模块
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    def forward(self, x):
        identity = self.shortcut(x)
        residual_out = self.residual_path(x)
        eca_out = self.eca(residual_out) # 对残差输出应用ECA
        out = F.relu(identity + eca_out)
        return out
# ==============================================================

# ==================== CBAM Modules ====================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x
# ==========================================================

# ==================== CBAM-ResNet 基础块 ====================
class CBAMBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(CBAMBasicBlock, self).__init__()
        self.residual_path = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.cbam = CBAM(planes) 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    def forward(self, x):
        identity = self.shortcut(x)
        residual_out = self.residual_path(x)
        cbam_out = self.cbam(residual_out) # 对残差输出应用CBAM
        out = F.relu(identity + cbam_out)
        return out
# ==============================================================

class PIDBlock(nn.Module):
    pass

# ===========================================================

class GenericNet(nn.Module):
    pass

def measure_inference_latency(model, device, input_size=(1, 3, 32, 32), batch_size=128, num_warmup=20, num_reps=100):
    print(f"\n--- 开始测量延迟: Batch Size={batch_size}, Reps={num_reps} ---")
    model.to(device)
    model.eval()
    dummy_input = torch.randn(batch_size, *input_size[1:], dtype=torch.float32).to(device)
    print("正在进行预热...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    print("预热完成。")
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = torch.zeros((num_reps,))
    print("正在进行计时测量...")
    with torch.no_grad():
        for i in range(num_reps):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            timings[i] = starter.elapsed_time(ender)
    total_time_ms = timings.sum().item()
    mean_time_per_batch_ms = timings.mean().item()
    std_time_per_batch_ms = timings.std().item()
    latency_per_image_ms = mean_time_per_batch_ms / batch_size
    throughput_images_per_sec = (num_reps * batch_size) / (total_time_ms / 1000)
    print("计时测量完成。")
    print(f"平均每批次延迟: {mean_time_per_batch_ms:.3f} ms (标准差: {std_time_per_batch_ms:.3f} ms)")
    return latency_per_image_ms, throughput_images_per_sec

def train_and_eval(model, model_name, trainloader, testloader, device, epochs=100, logger=None):
    print(f"\n--- 开始训练: {model_name} ---")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc, best_model_path = 0.0, ""
    total_start_time = time.time()
    save_dir = './Flowers5'
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss, train_correct_top1, train_correct_top5, train_total = 0.0, 0, 0, 0
        for data in trainloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            train_loss += loss.item()
            train_total += labels.size(0)
            _, pred_topk = outputs.topk(5, 1, True, True)
            correct = pred_topk.t().eq(labels.view(1, -1).expand_as(pred_topk.t()))
            train_correct_top1 += correct[:1].sum().item()
            train_correct_top5 += correct[:5].sum().item()

        train_acc_top1 = 100 * train_correct_top1 / train_total
        train_acc_top5 = 100 * train_correct_top5 / train_total
        avg_train_loss = train_loss / len(trainloader)
        
        model.eval()
        test_loss, test_correct_top1, test_correct_top5, test_total = 0.0, 0, 0, 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images); loss = criterion(outputs, labels)
                test_loss += loss.item()
                test_total += labels.size(0)
                _, pred_topk = outputs.topk(5, 1, True, True)
                correct = pred_topk.t().eq(labels.view(1, -1).expand_as(pred_topk.t()))
                test_correct_top1 += correct[:1].sum().item()
                test_correct_top5 += correct[:5].sum().item()

        test_acc_top1 = 100 * test_correct_top1 / test_total
        if test_acc_top1 > best_acc:
            best_acc = test_acc_top1
            best_model_path = os.path.join(save_dir, f"{model_name.replace(' ', '_')}_best.pth")
            state = {'net': model.state_dict(), 'acc': best_acc, 'epoch': epoch}
            torch.save(state, best_model_path)

        test_acc_top5 = 100 * test_correct_top5 / test_total
        avg_test_loss = test_loss / len(testloader)
        epoch_duration = time.time() - epoch_start_time
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        print(f'Epoch {epoch+1:>{len(str(epochs))}}/{epochs} | LR: {current_lr:.5f} | Train Loss: {avg_train_loss:.4f} | Test Acc Top-1: {test_acc_top1:.2f}% | Test Acc Top-5: {test_acc_top5:.2f}% (Best Top-1: {best_acc:.2f}%) | Time: {epoch_duration:.2f}s')

        if logger: logger.log([model_name, epoch + 1, current_lr, avg_train_loss, train_acc_top1, train_acc_top5, avg_test_loss, test_acc_top1, test_acc_top5, epoch_duration])
        
    total_training_time = time.time() - total_start_time
    print(f"最终 {model_name} 在测试集上的最佳Top-1准确率: {best_acc:.2f}%")
    print(f"总训练耗时: {total_training_time / 60:.2f} 分钟")
    if best_model_path: print(f"最佳模型已保存在: {best_model_path}")
    return best_acc, total_training_time

def run_experiment(gpu_id):
    SEED = 2025
    set_seed(SEED)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # --- 数据加载 ---
    BATCH_SIZE, NUM_CLASSES, DATASET = 32, 5, 'Flowers5'
    IMAGE_SIZE = 224
    print(f"\n--- 加载 {DATASET} 数据集 ---")
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE), 
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
        ])
    transform_test = transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 20),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    train_dir = './data/flowers/train'
    val_dir = './data/flowers/test'
    print(f"\n--- 加载 {DATASET} 数据集 ---")
    trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)
    testset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform_test)
    dummy_input_size = (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    g = torch.Generator(); g.manual_seed(SEED)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, generator=g)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print("数据集加载完成。")
    DATASET_SEED = 'Flowers5_SEED'
    # --- 日志记录 ---
    log_filename = f"{DATASET_SEED.lower()}_exp_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    log_headers = ["model_name", "epoch", "learning_rate", "train_loss", "train_acc_top1", "train_acc_top5", "test_loss", "test_acc_top1", "test_acc_top5", "epoch_duration_s"]
    main_logger = CSVLogger(filepath=f"logs_{DATASET_SEED.lower()}", filename=log_filename, headers=log_headers)
    
    # --- 实验控制中心 ---
    resnet34_blocks = [3, 4, 6, 3]
    # resnet34_blocks = [2, 2, 2, 2]
    def ResNet34_Baseline(num_classes=10):
        return GenericNet([MyBasicBlock]*4, resnet34_blocks, num_classes=num_classes, base_planes=66)

    def SEResNet34(num_classes=10):
        return GenericNet([SEBasicBlock]*4, resnet34_blocks, num_classes=num_classes, base_planes=66)

    def ECAResNet34(num_classes=10):
        return GenericNet([ECABasicBlock]*4, resnet34_blocks, num_classes=num_classes, base_planes=66)

    def CBAMResNet34(num_classes=10): 
        return GenericNet([CBAMBasicBlock]*4, resnet34_blocks, num_classes=num_classes, base_planes=66)
    
    def PIDANET(num_classes=200, base_planes=64, d_stages=[True, True, True, True]):
        # 根据 d_stages 列表决定每个 stage 使用哪种 block
        block_types = []
        for use_pid_block in d_stages:
            if use_pid_block:
                block_types.append(PIDBlock)
            else:
                block_types.append(MyBasicBlock)
        return GenericNet(block_types, num_blocks=resnet34_blocks, num_classes=num_classes, base_planes=base_planes)

    # --- 模型实例化 ---
    set_seed(SEED); baseline_model = ResNet34_Baseline(num_classes=NUM_CLASSES)
    set_seed(SEED); se_model = SEResNet34(num_classes=NUM_CLASSES)
    set_seed(SEED); eca_model = ECAResNet34(num_classes=NUM_CLASSES)
    set_seed(SEED); cbam_model = CBAMResNet34(num_classes=NUM_CLASSES)
    set_seed(SEED)  
    pid_model = PIDANET(num_classes=NUM_CLASSES, base_planes=64, d_stages=[False, False, False, True])

    # --- 模型复杂度对齐检查 ---
    try:
        from thop import profile
        dummy_input = torch.randn(*dummy_input_size).to(device)
        print("\n--- 模型复杂度对齐检查 ---")

        models_to_check = {
            "ResNet-34 Baseline": baseline_model,
            "SE-ResNet-34": se_model,
            "ECA-ResNet-34": eca_model,
            "CBAM-ResNet-34": cbam_model,
            "PID-ResNet-34": pid_model,
        }
        
        results = {}
        for name, model in models_to_check.items():
            model_on_device = model.to(device)
            flops, params = profile(model_on_device, inputs=(dummy_input,), verbose=False)
            results[name] = {'params': params, 'flops': flops}
            print(f"{name:<25}: {params/1e6:.2f}M Params, {flops/1e9:.2f}G FLOPs")
    except Exception as e:
        print(f"\n复杂度检查失败: {e}")
        results = {name: {'params': 0, 'flops': 0} for name in models_to_check.keys()}
    
    # --- 运行实验 ---
    num_epochs_for_run = 100
    
    # 训练并评估 Baseline
    # base_acc, base_time = train_and_eval(baseline_model, "ResNet34 Baseline", trainloader, testloader, device, epochs=num_epochs_for_run, logger=main_logger)
    # 训练并评估 SE-ResNet
    # se_acc, se_time = train_and_eval(se_model, "SE-ResNet34", trainloader, testloader, device, epochs=num_epochs_for_run, logger=main_logger)
    # eca_acc, eca_time = train_and_eval(eca_model, "ECA-ResNet34", trainloader, testloader, device, epochs=num_epochs_for_run, logger=main_logger)
    # cbam_acc, cbam_time = train_and_eval(cbam_model, "CBAM-ResNet34", trainloader, testloader, device, epochs=num_epochs_for_run, logger=main_logger)
    pid_acc, pid_time = train_and_eval(pid_model, "PID34(0001a)55", trainloader, testloader, device, epochs=num_epochs_for_run, logger=main_logger)
    
    # --- 测量延迟 ---
    inference_batch_size = BATCH_SIZE
    dummy_input_size_for_latency = (1, 3, 224, 224)
    # base_latency, base_throughput = measure_inference_latency(baseline_model, device, dummy_input_size_for_latency, inference_batch_size)
    # se_latency, se_throughput = measure_inference_latency(se_model, device, dummy_input_size_for_latency, inference_batch_size)
    # eca_latency, eca_throughput = measure_inference_latency(eca_model, device, dummy_input_size_for_latency, inference_batch_size)
    # cbam_latency, cbam_throughput = measure_inference_latency(cbam_model, device, dummy_input_size_for_latency, inference_batch_size)
    pid_latency, pid_throughput = measure_inference_latency(pid_model, device, dummy_input_size_for_latency, inference_batch_size)


    # --- 打印最终结果表格 ---
    print(f"\n\n--- 最终实验结果对比 (ResNet-34 on {DATASET}) ---")
    print("-" * 125)
    print(f"| {'模型':<25} | {'参数量 (M)':<12} | {'FLOPs (G)':<10} | {'延迟 (ms/img)':<15} | {'吞吐量 (img/s)':<17} | {'Top-1 准确率 (%)':<20} | {'总训练时间 (s)':<15} |")
    print(f"|{'-'*27}|{'-'*14}|{'-'*12}|{'-'*17}|{'-'*19}|{'-'*22}|{'-'*17}|")

    all_results = {
        # "ResNet-34 Baseline": (base_acc, base_time, base_latency, base_throughput),
        # "SE-ResNet-34": (se_acc, se_time, se_latency, se_throughput),
        # "ECA-ResNet-34": (eca_acc, eca_time, eca_latency, eca_throughput),
        # "CBAM-ResNet-34": (cbam_acc, cbam_time, cbam_latency, cbam_throughput),
        "PID-ResNet-34": (pid_acc, pid_time, pid_latency, pid_throughput)
    }

    for name, (acc, train_time, latency, throughput) in all_results.items():
        params = results[name]['params'] if name in results else 0
        flops = results[name]['flops'] if name in results else 0
        print(f"| {name:<25} | {params/1e6:<12.2f} | {flops/1e9:<10.2f} | {latency:<18.4f} | {int(throughput):<20} | {acc:<15.2f} | {train_time:<15.2f} |")
    print("-" * 128)

# --- 程序入口 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='在Flowers-5上进行ResNet-34, SE-ResNet-34和PID-ResNet-34的对比实验')
    parser.add_argument('--gpu', type=int, default=0, help='要使用的GPU ID')
    args = parser.parse_args()
    run_experiment(gpu_id=args.gpu)
