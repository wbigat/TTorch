"""两台模拟硬件：CudaHw（接口语义标准）与 XpuHw（接口非标）。

XpuHw 故意制造的非标 gap，考验 Platform adapter 的填补能力：
- setDevice 只接受 "XPU:n" 字符串，currentDevice 也返回字符串
- sync() 只等待当前 stream（不满足 device-wide 语义）
- 显存查询接口名不同（memUsed），且不支持 P2P（静态事实不同）
- profiler 标注接口名不同（profMark）
- 原生错误类型 XpuHwError（与 CudaHwError 不同，考验错误映射）
"""


class CudaHwError(Exception):
    """CUDA 栈原生错误。"""


class XpuHwError(Exception):
    """XPU 栈原生错误。"""


class GcuHwError(Exception):
    """GCU 栈原生错误。"""


class AsyncHandle:
    """异步提交句柄：wait() 后才算完成。"""

    def __init__(self, queue, work):
        self._queue = queue
        self._work = work
        self._done = False

    def wait(self):
        if not self._done:
            self._queue.remove(self._work)
            self._done = True


class MockStream:
    """mock 平台的原生 stream 对象（对标真实栈 torch.cuda.Stream：自带
    wait_stream 方法——那是框架 ddp fencing 的对象方法形态）。

    mock 硬件内部仍用 int sid 记账；本对象 __hash__/__eq__ 与裸 sid 兼容，
    使 streams dict / wait_edges / current_stream 的 int 语义不受影响。
    存在理由：TT 门面不再包装 stream（无 TTStream），各平台原生对象须自带
    框架要用的方法，mock 的"原生对象"因此从裸 int 升级为本类。
    """

    def __init__(self, sid, hw):
        self.sid = sid
        self._hw = hw

    def wait_stream(self, other):   # 框架侧 fencing 形态（对象方法）
        # 模拟"原生 stream 的原生方法"（对标 torch.cuda.Stream.wait_stream，
        # 真实栈不经 TT adapter）。mock 硬件的跨流等待物理含义 = wait_edges
        # 留痕，直接记录——与 CudaHw.wait_stream / XpuHw.streamWait 的接口名
        # 差异无关（那是实现类填 gap 的事，非对象方法的事）。
        self._hw.wait_edges.append((self, other))

    def __eq__(self, o):
        return self.sid == (o.sid if isinstance(o, MockStream) else o)

    def __hash__(self):
        return hash(self.sid)

    def __repr__(self):
        return f"<MockStream:{self.sid}>"


# 模拟"当前机器上插了什么卡"。真实环境的对应物是驱动/SO/设备节点的存在性
# 探测（只看不初始化）；demo 用这个集合模拟。本场景是 xpu-only 机器。
PRESENT_HARDWARE = {"xpu"}


class CudaHw:
    """语义与 CUDA 文档一致：synchronize 为 device-wide 等待。"""

    p2p_supported = True   # 静态平台事实
    sdk_version = "12.4"   # 驱动/runtime 版本（供版本协商）

    @classmethod
    def probe_present(cls):
        """存在性探测：只看不初始化，供 Platform Manager 自动检测。

        注意：GCU 栈会冒充 CUDA（torch.cuda.is_available() 报 True，
        CUDA API 有响应），所以 gcu 在场时本探测也会命中——这是
        compat claim，不是 native 证据，靠 GcuHw.probe_present +
        masquerades_cuda 降级机制纠正。
        """
        return "cuda" in PRESENT_HARDWARE or "gcu" in PRESENT_HARDWARE

    def __init__(self, device_count=8):
        self._current = 0
        self._count = device_count
        self._used = 0  # 动态显存水位（字节）
        self.marks = []  # profiler 标注记录
        self.streams = {0: [], 1: [], 2: []}  # stream_id -> pending work
        self._next_stream = 3
        self.current_stream = MockStream(0, self)  # 当前 stream（原生对象）
        self.wait_edges = []           # 跨流依赖留痕 [(waiter, waitee)]
        self._rng = "cuda-rng-seed-0"  # RNG 状态（契约：不透明 str）
        self.total_mem = 80 * 1024**3  # 物理显存 80GB

    def set_device(self, idx):
        if not 0 <= idx < self._count:
            raise CudaHwError(f"invalid device ordinal {idx}")
        self._current = idx

    def current_device(self):
        return self._current

    def device_count(self):
        return self._count

    def create_stream(self):
        """运行期动态新建 stream——adapter 的 device-wide 等待必须覆盖它。"""
        sid = self._next_stream
        self._next_stream += 1
        self.streams[sid] = []
        return MockStream(sid, self)

    def submit(self, stream_id, work):
        self.streams[stream_id].append(work)

    def submit_async(self, stream_id, work):
        self.submit(stream_id, work)
        return AsyncHandle(self.streams[stream_id], work)

    def sync_device(self):
        for q in self.streams.values():
            q.clear()

    def capture(self, fn):
        """CUDA Graph 捕获（模拟：直接执行并留痕）。"""
        self.marks.append("graph-capture")
        return fn()

    def alloc(self, nbytes):
        self._used += nbytes

    def free(self, nbytes):
        self._used -= nbytes

    def memory_allocated(self):
        return self._used

    def annotate(self, name):
        self.marks.append(name)

    def pending(self, stream_id):
        return len(self.streams[stream_id])

    # --- P0 扩展：stream 原语 / RNG / 显存信息 / allocator（CUDA 侧语义标准） ---

    def set_stream(self, sid):
        self.current_stream = sid

    def wait_stream(self, waiter, waitee):
        self.wait_edges.append((waiter, waitee))

    def get_rng_state(self):
        return self._rng

    def set_rng_state(self, state):
        if not isinstance(state, str):
            raise CudaHwError(f"bad rng state type {type(state).__name__}")
        self._rng = state

    def mem_get_info(self):
        return (self.total_mem - self._used, self.total_mem)

    def empty_cache(self):
        self.marks.append("empty_cache")


class GcuHw(CudaHw):
    """GCU 栈：接口 CUDA 兼容（继承 CudaHw 语义），但有原生探测通道。

    真实对应物：GCU/MACA 这类栈会让 torch.cuda.is_available() 报 True，
    只看 CUDA API 存在性会误判成 NVIDIA。原生探测走厂商私有通道
    （设备节点 / 厂商 SO），demo 用 PRESENT_HARDWARE 模拟。
    """

    sdk_version = "4.2"  # GCU 自家 SDK 版本线，与 CUDA 版本无关

    @classmethod
    def probe_present(cls):
        """native 探测：厂商私有通道，不会被别家冒充。"""
        return "gcu" in PRESENT_HARDWARE


class XpuHw:
    """非标接口：字符串设备名 + sync 只等当前 stream + 显存/标注接口名不同。"""

    p2p_supported = False  # 静态平台事实（与 CUDA 不同）
    sdk_version = "2.5"

    @classmethod
    def probe_present(cls):
        return "xpu" in PRESENT_HARDWARE

    def __init__(self, device_count=8):
        self._current = 0
        self._count = device_count
        self._used = 0
        self.marks = []
        self.streams = {0: [], 1: [], 2: []}
        self._next_stream = 3
        self.current_stream = MockStream(0, self)
        self.wait_edges = []              # 跨流依赖留痕
        self._rng = "xpu-rng-seed-0"      # RNG 状态（契约：不透明 str）
        self.total_mem_mib = 80 * 1024    # 物理显存 80GB（非标：MiB 单位）

    def setDevice(self, name):  # 非标：只接受 "XPU:n"
        if not name.startswith("XPU:"):
            raise XpuHwError(f"bad device name {name!r}, expect 'XPU:n'")
        idx = int(name.split(":")[1])
        if not 0 <= idx < self._count:
            raise XpuHwError(f"device {idx} out of range")
        self._current = idx

    def currentDevice(self):  # 非标：返回字符串
        return "XPU:%d" % self._current

    def deviceCount(self):
        return self._count

    def create_stream(self):
        sid = self._next_stream
        self._next_stream += 1
        self.streams[sid] = []
        return MockStream(sid, self)

    def submit(self, stream_id, work):
        self.streams[stream_id].append(work)

    def submit_async(self, stream_id, work):
        self.submit(stream_id, work)
        return AsyncHandle(self.streams[stream_id], work)

    def set_stream(self, s):
        self.current_stream = s

    def sync(self):  # gap：只等当前 stream，不是 device-wide
        self.streams[self.current_stream].clear()

    def alloc(self, nbytes):
        self._used += nbytes

    def free(self, nbytes):
        self._used -= nbytes

    def memUsed(self):  # 非标：接口名不同
        return self._used

    def profMark(self, name):  # 非标：接口名不同
        self.marks.append(name)

    def pending(self, stream_id):
        return len(self.streams[stream_id])

    # --- P0 扩展（XPU 侧全部非标，adapter 的填 gap 对象） ---

    def streamWait(self, waiter, waitee):  # 非标名
        self.wait_edges.append((waiter, waitee))

    def rngState(self):  # 非标名
        return self._rng

    def setRngState(self, state):  # 非标名
        if not isinstance(state, str):
            raise XpuHwError(f"bad rng state type {type(state).__name__}")
        self._rng = state

    def memInfo(self):  # 非标：dict 结构 + MiB 单位 + _used 是字节需换算
        free_mib = self.total_mem_mib - self._used // 1024**2
        return {"freeMiB": free_mib, "totalMiB": self.total_mem_mib}

    def purge(self):  # 非标名（empty_cache 的对应物）
        self.marks.append("purge")
