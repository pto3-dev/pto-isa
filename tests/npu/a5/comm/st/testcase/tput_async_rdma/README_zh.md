# RDMA异步ST（HNS1825目标网卡平台）

本目录包含RDMA异步ST的共享实现。`tput_async_rdma` 验证远程WRITE，`tget_async_rdma` 验证远程READ，
`tput_async_notify_rdma` 验证带`Set`通知的远程WRITE；三个目标复用同一套RDMA测试Kernel实现。

## 前置条件

- 配备HNS1825网卡及匹配驱动、HCOMM组件的A5环境
- 按顶层[测试说明](../../../../../../README_zh.md)配置MPI和HCCL
- 各参与rank使用的RDMA网卡IPv4互相可达

## 构建与运行

在CMake配置前选择RDMA实现，再运行所需的RDMA ST target：

```bash
export PTO_RDMA_BACKEND=HNS_1825
python3 tests/script/run_st.py -r npu -v a5 -t comm/tput_async_rdma -d -n 2
python3 tests/script/run_st.py -r npu -v a5 -t comm/tget_async_rdma -d -n 2
python3 tests/script/run_st.py -r npu -v a5 -t comm/tput_async_notify_rdma -d -n 2
```

首次验证`TPUT_ASYNC_NOTIFY`时，先只运行2个rank的定向用例：

```bash
export PTO_RDMA_BACKEND=HNS_1825
python3 tests/script/run_st.py -r npu -v a5 -t comm/tput_async_notify_rdma \
    -g TPutAsyncNotifyRdma.Int32SetAndCanaries -d -n 2
```

该用例检查远端payload、远端signal及signal两侧canary，并等待接口返回的`AsyncEvent`完成。接收端观察到
signal后先维护数据缓存，再校验payload。当前RDMA后端不支持`AtomicAdd`，因此没有对应的RDMA用例。

`PTO_RDMA_BACKEND` 仅在CMake配置该ST构建时读取。未设置、空值或不支持的值会构建不含RDMA支持的测试。`run_st.py` 默认重新构建；修改该变量后，不要使用 `-w/--without-build` 复用已有二进制。

## 端点发现

Bootstrap先解析每个rank的物理设备id和RDMA IPv4，再通过MPI交换端点与注册内存信息。本地IPv4的查找顺序为：

1. 固定 `/etc/hccl_rootinfo.json` 中与物理设备匹配的CLOS IPv4。
2. 由HCOMM topology组件解析固定 `/var/run/ascend-topologyd/virtualTopology.xml`。
3. 使用下表中的测试专用IP变量兜底。

ST不会生成或修改这两个拓扑文件，也不提供路径覆盖变量。

| 变量 | 说明 |
|---|---|
| `PTO_RDMA_BACKEND` | 配置阶段选择项，当前唯一支持值为 `HNS_1825`。|
| `PTO_ROCE_PHYIDS` | 可选，按MPI rank索引、逗号分隔的物理设备id。|
| `PTO_ROCE_LOCAL_IP` | 当前MPI进程使用的最终兜底IPv4；必要时需为各rank分别设置。|
| `PTO_ROCE_IPS` | 最终兜底列表，按MPI rank排序且IPv4数量必须等于rank数。|
| `PTO_ROCE_BASE_PORT` | 各rank一致的channel base port，默认 `60032`。|
| `PTO_ROCE_VERBOSE` | 设为 `1`，打印端点、MR、channel和释放进度。|
| `HCCL_RDMA_TC` | HCOMM traffic class，默认 `132`。|
| `HCCL_RDMA_SL` | HCOMM service level，默认 `4`。|

`PTO_ROCE_LOCAL_IP` 的优先级高于 `PTO_ROCE_IPS`；root-info或virtual topology解析成功时，两者均被忽略。所有rank必须使用相同的base port；使用 `PTO_ROCE_IPS` 时，还必须使用相同的按rank排序列表。

## 问题定位

- CMake提示RDMA未使能时，设置 `PTO_RDMA_BACKEND=HNS_1825`，并在不使用 `-w` 的情况下重新配置。
- 端点发现失败时，检查物理设备映射以及root-info或virtual topology中的CLOS IPv4，必要时使用测试专用IP变量兜底。
- HCOMM无法从默认路径加载HNS1825 verbs provider时，将 `IBV_EXTEND_DRIVERS` 指向驱动提供的 `libhrn5-rdmav34.so`。
- 设置 `PTO_ROCE_VERBOSE=1`，可区分端点发现、MR注册、channel建链和释放阶段的错误。
