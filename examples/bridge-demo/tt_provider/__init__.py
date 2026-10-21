"""TT Provider 垂直切片：性能增量层，与 Bridge 功能层分离。

定位（定稿）：Provider 只影响性能——某 platform 没实现某 provider、或
provider 被 switch 禁用，都不会导致功能问题，系统退回 reference
（torch-native 基线）仍正确训练。

组成：
- provider_mgr.py    Provider Manager：registry / selection / 功能&精度
                     准入门禁 / switch 启用控制 / 绑定缓存 / 分发回退
- provenance.py      一等能力：选择路径、实际执行参数、性能数据、回退事实
- implementations.py 三类实现：reference（native 基准）、Fused 算子
                     （AOT：厂商融合 / ABO 融合；JIT）、可选 schedule 插槽
"""
