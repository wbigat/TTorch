"""真实框架替身：tiny decoder-only LLM 的 SFT 主链路（模拟 swift 调用形态）。

设备交互全部走 torch.cuda.* 属性调用——运行期被 cuda-compat 绑定到
bridge 门面，本文件源码一行不改地跑在 CUDA / XPU(masquerade/native) /
CPU(dev) 上。唯一的可移植性约定：设备构造走 bridge 的 device_spec()
门面——torch.device("cuda:%d") 字面量是 C 类盘点面，WRAP 决策落地前
的过渡形态（masquerade 栈上字面量恰好能用，native xpu 上必须 device_spec）。
"""
import torch
from torch import nn

from tt_bridge import platform_mgr
from tt_bridge.platform_adapter import TTPlatformError

VOCAB, DIM, HEADS, LAYERS, SEQ, BATCH = 512, 256, 4, 2, 64, 8


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):  # x: (B, S, D)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = RMSNorm(DIM)
        self.qkv = nn.Linear(DIM, DIM * 3, bias=False)
        self.proj = nn.Linear(DIM, DIM, bias=False)
        self.n2 = RMSNorm(DIM)
        self.mlp = nn.Sequential(nn.Linear(DIM, DIM * 4), nn.GELU(),
                                 nn.Linear(DIM * 4, DIM))

    def forward(self, x):  # (B, S, D)
        h = self.n1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        B, S, _ = x.shape
        q = q.view(B, S, HEADS, -1).transpose(1, 2)  # (B, H, S, d)
        k = k.view(B, S, HEADS, -1).transpose(1, 2)
        v = v.view(B, S, HEADS, -1).transpose(1, 2)
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(B, S, DIM)
        x = x + self.proj(a)
        return x + self.mlp(self.n2(x))


class TinyLLM(nn.Module):
    """~2M 参数 decoder-only：参数量 = emb 512*256 + 2*block + head 512*256。"""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DIM)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.norm = RMSNorm(DIM)
        self.head = nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, ids):  # (B, S) -> (B, S, V)
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


def make_batch(step, device):
    """专用 Generator 固定种子：跨设备/跨运行数据逐位一致（精度对账前提）。"""
    g = torch.Generator().manual_seed(1000 + step)
    x = torch.randint(0, VOCAB - 1, (BATCH, SEQ), generator=g)
    y = (x + 1) % VOCAB  # 确定性可学习映射
    return x.to(device), y.to(device)


def _bridge_io(step):
    """swift 主链路形态的设备交互：全部走 torch.cuda.* 属性调用
    （运行期已被 cuda-compat 绑到 bridge 门面）。"""
    if step % 10 == 0:
        # checkpoint 形态：RNG 存取回环（megatron_lm_utils.py:107/599）
        state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(state)
    if platform_mgr.capability("stream_api"):
        # ddp 侧流 fencing 形态（megatron_lm_utils.py:686-704）：双向依赖边
        side = torch.cuda.Stream()
        main = torch.cuda.current_stream()
        side.wait_stream(main)
        with torch.cuda.stream(side):
            pass  # 梯度 bucket 通信占位
        main.wait_stream(side)
    torch.cuda.synchronize()


def sft_run(device, steps=50, precision="fp32", seed=7, bridged=True):
    """SFT 训练 loop。返回 (losses, eval_logits_cpu, mem_info)。

    模型在 CPU 以固定种子初始化后 .to(device)，保证跨设备权重逐位一致
    （精度对账前提）。bridged=True 时逐步执行 _bridge_io 设备交互。
    """
    torch.manual_seed(seed)
    model = TinyLLM().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    use_bf16 = precision == "bf16"

    mem_info = None
    if bridged:
        try:
            free, total = torch.cuda.mem_get_info()  # patcher max_memory 形态
            mem_info = (free, total)
        except TTPlatformError as e:
            mem_info = f"degraded: {e}"  # 能力驱动降级：平台声明 memory_query=False

    losses = []
    for step in range(steps):
        x, y = make_batch(step, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if bridged:
            _bridge_io(step)
        losses.append(float(loss))
    if bridged:
        torch.cuda.empty_cache()

    model.eval()
    with torch.no_grad():
        xe, _ = make_batch(99999, device)
        eval_logits = model(xe).float().cpu()
    return losses, eval_logits, mem_info, n_params
