<p align="center">
  <img src="figures/pto_logo.svg" alt="PTO Tile Lib" width="220" />
</p>

# 快速开始

本指南帮助您设置和运行 PTO ISA 项目。它涵盖两种主要场景：

1. **CPU 模拟器**（推荐新手）- 跨平台支持 macOS、Linux 和 Windows
2. **NPU 环境**（高级）- Linux 上的 Ascend A2/A3(910B/910C) 与 CANN toolkit

请选择符合您需求的部分。大多数用户应该从 CPU 模拟器开始。

---

## 第一部分：CPU 模拟器（跨平台）

CPU 模拟器是最简单的入门方式。它可以在 macOS、Linux 和 Windows 上运行，无需专用硬件。

### 先决条件

**必需项：**
- Git
- Python `>= 3.11`
- CMake `>= 3.16`
- 支持 C++20 的 C++ 编译器：
  - Linux: GCC 13+ 或 Clang 15+（GCC >= 14 启用 bfloat16 支持）
  - macOS: Xcode/AppleClang（或 Homebrew LLVM）
  - Windows: Visual Studio 2022 Build Tools (MSVC)
- Python 包：`numpy >= 1.22.0`

`tests/run_cpu.py` 可以自动安装 `numpy`（除非您传递 `--no-install` 参数）。

**可选项（用于加速构建）：**
- Ninja (CMake 生成器)
- 互联网连接（如果未在系统范围内安装 GoogleTest，CMake 可能需要获取它）

### 操作系统特定设置

#### macOS

安装 Xcode 命令行工具：

  ```bash
  xcode-select --install
  ```

安装依赖项（推荐通过 Homebrew）：

  ```bash
  brew install cmake ninja python
  ```

如果不使用 Homebrew，请确保 `python3`、`cmake` 和现代的 `clang++` 在 `PATH` 环境变量中。

#### Linux (Ubuntu 20.04+)

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build python3 python3-pip python3-venv git
```

#### Windows

安装以下软件：
- Git for Windows
- Python 3（并确保其在 `PATH` 中）
- CMake
- Visual Studio 2022 Build Tools（需包含"使用 C++ 的桌面开发"）

使用 `winget`（可选）：

```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.11 -e
winget install --id Kitware.CMake -e
winget install --id Microsoft.VisualStudio.2022.BuildTools -e
```

安装完成后，打开 **Developer Command Prompt for VS 2022**（或确保 `cl.exe` 在 `PATH` 中）。

**替代方案：手动安装编译器**

如果您不想使用 Visual Studio，可以通过下列方式之一手动安装GCC/Clang编译器：
- [WinLibs](https://winlibs.com)
- [MSYS2](https://www.msys2.org)

安装后，将 `{COMPILER_INSTALLATION_PATH}/bin` 目录添加到 `PATH` 环境变量中（在 PowerShell 中使用 `gcc -v/clang -v` 验证）。

### 获取代码

```bash
git clone https://gitcode.com/cann/pto-isa.git
cd pto-isa
```

### Python 环境设置

创建并激活虚拟环境：

**macOS / Linux：**

  ```bash
  python3 -m venv .venv-mkdocs
  source .venv-mkdocs/bin/activate
  python -m pip install -U pip
  python -m pip install numpy
  ```

**Windows (PowerShell)：**

  ```powershell
  py -3 -m venv .venv-mkdocs
  .\.venv-mkdocs\Scripts\Activate.ps1
  python -m pip install -U pip
  python -m pip install numpy
  ```

### 运行 CPU 模拟器

构建并运行 `tests/cpu/st` 目录下的所有 CPU ST 测试二进制文件：

```bash
python3 tests/run_cpu.py --clean --verbose
```

**常用选项：**

运行单个测试用例：

  ```bash
  python3 tests/run_cpu.py --testcase tadd
  ```

运行特定的 gtest 用例：

  ```bash
  python3 tests/run_cpu.py --testcase tadd --gtest_filter 'TADDTest.*'
  ```

构建并运行 GEMM 演示：

  ```bash
  python3 tests/run_cpu.py --demo gemm --verbose
  ```

构建并运行 Flash Attention 演示：

  ```bash
  python3 tests/run_cpu.py --demo flash_attn --verbose
  ```

**其他选项：**

指定编译器路径：

  ```bash
  python3 tests/run_cpu.py --cxx=/path/to/compiler
  ```

打印详细日志：

  ```bash
  python3 tests/run_cpu.py --verbose
  ```

删除构建目录并重新构建：

  ```bash
  python3 tests/run_cpu.py --clean
  ```

Windows 特定选项（如需要）：

  ```bash
  python3 tests/run_cpu.py --clean --generator "MinGW Makefiles" --cmake_prefix_path D:\gtest\
  ```

设置库路径（Linux）：

  ```bash
  export LD_LIBRARY_PATH=/path_to_compiler/lib64:$LD_LIBRARY_PATH
  ```

---

## 第二部分：NPU 环境（Ascend 910B/910C，仅限 Linux）

本部分适用于需要在 Ascend NPU 硬件或模拟器上运行的用户。它需要 Linux 环境和 Ascend CANN toolkit。

### 说明

根目录 `README_zh.md` 仅保留最短上手路径与常用命令；如果您需要更完整的运行、测试与脚本说明，可进一步参考：

- [测试说明](../tests/README_zh.md)
- [文档构建说明](website_zh.md)

### 先决条件

**系统要求：**
- Linux（推荐 Ubuntu 20.04+）
- Python >= 3.9.x
- GCC >= 7.3.0
- CMake >= 3.16.0

> **备注：** Python 宣布 3.7.x/3.8.x 已经 EOL，CANN 即将停止对该版本的支持，请升级到 Python >= 3.9.x 的版本。

**GoogleTest（单元测试所需）：**

下载 [GoogleTest 1.14.0](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz) 并安装：

```bash
tar -xf googletest-1.14.0.tar.gz
cd googletest-1.14.0
mkdir temp && cd temp
cmake .. -DCMAKE_CXX_FLAGS="-fPIC"
make
sudo make install
```

> **注意：** Python 需要以下包：os、numpy、ctypes、struct、copy、math、enum、ml_dtypes 等。
>
> 如果您使用不同的标志安装了 GoogleTest（例如 `-D_GLIBCXX_USE_CXX11_ABI=0`），则必须在 `tests/npu/[a2a3|a5]/src/st/CMakeLists.txt` 中添加 `add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)` 进行相应更新。

**CANN Toolkit 安装**

- Ascend NPU 驱动和固件（用于硬件执行）
- CANN toolkit >= 8.5.0

> **重要提示：** 请确保在安装完 CANN toolkit 后，执行以下命令设置环境变量：
>
> ```bash
> source /usr/local/Ascend/cann/bin/setenv.bash
> ```
>
> 然后验证 bisheng 的 GCC 版本与系统 GCC 版本兼容或一致：
>
> ```bash
> bisheng -v
> gcc -v
> ```
>
> 如果版本不同，您可以在 bisheng -v 所示的 gcc 路径下安装与 gcc -v 显示版本一致的 GCC，或将默认的 gcc 替换为与 bisheng -v 显示一致的版本。

### 安装选项

#### 选项 1：快速安装（推荐）

完整的安装指南（包括驱动、固件和 toolkit）：

<https://www.hiascend.com/cann/download>

此方法会自动处理所有依赖项。

#### 选项 2：手动安装

**步骤 1：安装驱动和固件**

在实际 NPU 硬件上运行时需要（如果仅构建或使用模拟器，可跳过）。

安装指南：[NPU 驱动和固件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_0005.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)

**步骤 2：安装 CANN Toolkit**

从 [CANN 下载页面](https://www.hiascend.com/developer/download/community/result?module=cann) 下载相应的 `Ascend-cann-toolkit_${cann_version}_linux-${arch}.run` 安装包。

要求版本：CANN >= 8.5.0

```bash
# 使安装包可执行
chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run

# 安装
./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --force --install-path=${install_path}
```

参数说明：
- `${cann_version}`：CANN toolkit 版本
- `${arch}`：CPU 架构（`aarch64` 或 `x86_64`）
- `${install_path}`：安装路径（可选）

默认路径：
- Root 用户安装：`/usr/local/Ascend/cann`
- 非 root 用户安装：`$HOME/Ascend/cann`

#### 选项 3：一键环境安装脚本

如果您希望一次性完成 CANN toolkit、PTO ISA 包和 GoogleTest 的安装，可使用项目自带的 `install_pto.sh` 脚本。脚本会自动检测 CANN toolkit 是否已安装：若已安装则直接安装 PTO ISA 包；若未安装则需通过第二个参数提供 CANN toolkit 安装包路径。

```bash
chmod +x ./scripts/install_pto.sh
./scripts/install_pto.sh <cann_toolkit_install_path> [cann_toolkit_package_path]
```

参数说明：
- `<cann_toolkit_install_path>`：CANN toolkit 的安装路径（用于检测是否已安装）
- `[cann_toolkit_package_path]`：CANN toolkit 安装包路径（可选；仅在 CANN toolkit 未安装时需要提供）

### 环境变量

在运行 NPU 测试之前设置 CANN 环境：

**Root 用户安装（默认路径）：**

```bash
source /usr/local/Ascend/cann/bin/setenv.bash
```

**非 root 用户安装（默认路径）：**

```bash
source $HOME/Ascend/cann/bin/setenv.bash
```

**自定义安装路径：**

```bash
source ${install_path}/cann/bin/setenv.bash
```

### 下载源代码

```bash
git clone https://gitcode.com/cann/pto-isa.git
cd pto-isa
```

### 运行 NPU 测试

**运行单个 ST 测试用例：**

  ```bash
  python3 tests/script/run_st.py -r [sim|npu] -v [a3|a5] [-a] -t [TEST_CASE] -g [GTEST_FILTER_CASE]
  ```

注意：`a3` 后端覆盖 A2/A3 系列（`include/pto/npu/a2a3`）；`-a`使能auto模式。

  示例：

  ```bash
  python3 tests/script/run_st.py -r npu -v a3 -t tmatmul -g TMATMULTest.case1
  python3 tests/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULTest.case1
  ```

**运行推荐的测试套件：**

  ```bash
# 从项目根目录执行
  chmod +x ./tests/run_st.sh
  ./tests/run_st.sh --a5 --npu --simple

# 对于模拟器（首先增加文件描述符限制）
ulimit -n 65536
./tests/run_st.sh --a3 --sim --all
  ```

**运行完整的 ST 测试：**

  ```bash
  chmod +x build.sh
  ./build.sh --run_all --a3 --sim
  ```

**运行简化的 ST 测试：**

  ```bash
  chmod +x build.sh
  ./build.sh --run_simple --a5 --npu
  ```

**打包：**

  ```bash
  chmod +x build.sh
  ./build.sh --pkg
  ```

  构建完成后，`.run` 安装包生成在 `scripts/package/output/` 目录下。

**安装：**

  通过 `.run` 自解压包安装，需指定安装类型（三选一）：

  | 参数 | 说明 |
  |------|------|
  | `--full` | 完整安装，包含头文件、库、测试资源等全部内容 |
  | `--run` | 运行时安装，仅安装运行所需的库和依赖 |
  | `--devel` | 开发环境安装，包含头文件和库 |

  ```bash
  # 完整安装到默认路径（需 root 权限）
  ./scripts/package/output/pto_isa_*.run --full

  # 安装到指定路径（无需 root）
  ./scripts/package/output/pto_isa_*.run --full --install-path=/your/install/path

  # 静默安装，跳过交互确认（适用于 CI/CD 等非交互环境）
  ./scripts/package/output/pto_isa_*.run --full --quiet

  # 仅安装运行时组件
  ./scripts/package/output/pto_isa_*.run --run --install-path=/your/install/path

  # 开发环境安装
  ./scripts/package/output/pto_isa_*.run --devel --install-path=/your/install/path --quiet
  ```

  常用安装参数说明：

  | 参数 | 说明 |
  |------|------|
  | `--install-path=<path>` | 指定安装目标目录 |
  | `--quiet` | 静默模式，跳过交互确认（适用于非交互环境） |
  | `--install-for-all` | 为所有用户安装 |
  | `--uninstall` | 卸载已安装的产品 |
  | `--upgrade` | 升级已有安装 |
  | `--version` | 查看包版本信息 |
  | `--pre-check` | 安装前预依赖检查 |

  更多参数可通过 `--help` 查看：

  ```bash
  ./scripts/package/output/pto_isa_*.run --help
  ```

---

## 下一步

- CPU 开发：探索 `tests/cpu/demos/` 下的演示
- NPU 开发：查看 `tests/npu/` 下的测试用例
- 查看 API 文档了解详细的指令使用方法
- 加入社区获取支持和讨论

## 实验性 A1 架构识别

CCEC `--cce-aicore-arch=dav-m200` 定义 `__NPU_ARCH__=2002`。公共头文件现在将其识别为
`ChipArch::A1` / `PTO_NPU_ARCH_A1`，不再使用能力全部开放的 `UNKNOWN` 配置。
A1 提供 FP32 和 INT32 累加器；BF16、FP8/FP4、跨核 SyncAll/通信、量化、直方图和 MX 布局能力仍禁用。
现有架构的枚举值保持不变。

目标识别和能力检查独立于指令覆盖率。下文描述已实现的 FP16/FP32 向量子集，
编译器降低与运行时集成仍须分别适配；不会启用 A2/A3 指令回退。

无需执行 NPU 内核即可在主机验证该配置：

```bash
clang++ -std=c++20 -D__CPU_SIM -D__NPU_ARCH__=2002 -I include \
  tests/compile/a1_architecture.cpp -o /tmp/pto-a1-architecture
/tmp/pto-a1-architecture
```

使用已安装的 CANN 工具链验证编译器实际目标宏：

```bash
ccec -c -std=c++17 -x cce --cce-aicore-only --cce-aicore-arch=dav-m200 \
  -I include tests/compile/a1_architecture.cpp -o /tmp/pto-a1-architecture.o
```

### A1 tile 分配预算

A1 适配初期限制 UB 分配预算为 128 KiB、L1 分配预算为 512 KiB。这是保守预算，
不是物理容量声明：CANN 的 310P3 平台配置报告 UB 为 256 KiB、L1 为 1 MiB。
L0A/L0B 各为 64 KiB，L0C 为 256 KiB。Bias、缩放缓冲及 scale tile 的分配预算保持为零，
直到相应路径验证完成。使用旧 184 KiB UB scratch 偏移的 A1 helper 仍未启用，即使用户分配范围低于该区域。

`tests/compile/a1_tile_buffers.cpp` 检查这些默认值并构造 16 x 128 的 FP16 向量 tile。
使用上面的主机/CCEC 命令，替换源文件名即可验证。此测试仅检查 tile 元数据和分配限制，
不启动指令，也不进行数值测试。

设备编译器还检查实际 TASSIGN 分配边界：4 KiB tile 放在 124 KiB 偏移处应成功。
增加 `-DPTO_A1_TEST_OVERFLOW` 会将分配再偏移 32 字节，编译必须以 `SA-0353` 拒绝。

### A1 核内事件

A1 同步限定在耦合核心内部。load/add/store 对应 MTE2/V/MTE3；Acc-to-Vec 对应 V，
Vec-to-Mat 对应 MTE3。不支持的直接累加器 store 保留未知 pipe 标记，不声明 FIX pipe 路径。
自动事件编号使用 0 到 6，ID 7 留给 ISA helper。显式 ID 7、跨核 ID 和 FFTS 地址在编译时拒绝。

`tests/compile/a1_local_events.cpp` 使用 CCEC 编译本地 set/wait/barrier 序列。
主机测试让编号循环 100 次，确认不会返回 ID 7。`PTO_A1_TEST_RESERVED_ID` 和
`PTO_A1_TEST_CROSS_CORE` 选择负向编译测试。上述检查不代表 NPU 同步正确性已验证；
实际指令执行仍须单独验收。

### A1 手动 FP16/FP32 向量指令

A1 的 `pto/pto-inst.hpp` 提供 `TASSIGN`、本地 `TSYNC`/`Event`、`TLOAD`、`TADD` 和 `TSTORE`。
向量指令要求 FP16 或 FP32、连续 row-major Vec tile、静态完整 valid shape、32 字节对齐的行和匹配的静态 ND tensor。
GM 指针必须按 32 字节对齐。`TADD` 使用 256 字节 repeat（128 个 FP16 或 64 个 FP32 元素），最后一个不足完整重复的部分使用掩码，超过 255 个 repeat 时拆分为多条指令。
复用的地址赋值 helper 不包含 A2/A3 设备指令。

暂不提供自动分配/同步、其他数据类型、动态或部分 valid shape、非对齐 DMA、atomic、unit flag、
累加器 store 及其他指令。使用 `TASSIGN<Addr>(tile)` 检查静态分配边界；调用者负责 tile 生命周期。缓冲须互不重叠或完全别名，不支持部分地址重叠。
PyPTO 执行需要配套的编译器与运行时特性；此 ISA 子集不提供 persistent runtime。

烟测内核加载两个 tensor，通过本地事件依赖执行加法并存储结果；CPU reference 使用同步 CPU API：

```bash
clang++ -std=c++20 -D__CPU_SIM -D__NPU_ARCH__=2002 -I include \
  tests/compile/a1_fp16_vector.cpp -o /tmp/pto-a1-vector-cpu
/tmp/pto-a1-vector-cpu
```

激活 CANN 环境后，使用已安装的头文件/库构建单入口设备内核与独立 host launcher：

```bash
ccec -c -std=c++17 -O2 -x cce --cce-aicore-only --cce-aicore-arch=dav-m200 \
  -I include tests/compile/a1_fp16_vector.cpp -o /tmp/pto-a1-vector.o
CANN_ARCH_DIR="$ASCEND_HOME_PATH/aarch64-linux"
g++ -std=c++17 -O2 -I "$CANN_ARCH_DIR/include" -I "$CANN_ARCH_DIR/pkg_inc" \
  -I "$CANN_ARCH_DIR/pkg_inc/runtime" tests/compile/a1_single_launch.cpp \
  -L "$CANN_ARCH_DIR/lib64" -lascendcl -lruntime -o /tmp/pto-a1-launch
```

先预约空闲物理设备；本次进程设置 `ASCEND_RT_VISIBLE_DEVICES` 后传入映射得到的逻辑编号。
主机有设备队列时应使用队列。例如物理设备 4 已预约：

```bash
ASCEND_RT_VISIBLE_DEVICES=4 /tmp/pto-a1-launch /tmp/pto-a1-vector.o 0
```

默认用例将 2048 个 FP16 结果与可精确表示的小整数加法结果比较。
增加 `-DPTO_A1_TEST_LARGE` 编译内核，并在启动时增加 `32768` 参数，可验证 256 个 repeat、
255-repeat 分段边界及 UB 就地加法。两个 64 KiB 输入缓冲恰好落在保守的 128 KiB UB 预算内。
先用同一宏编译并运行 CPU reference。launcher 仅管理自己创建的 context 和 stream，不 reset 设备。

`PTO_A1_TEST_NON_FLOAT`、`PTO_A1_TEST_DYNAMIC_VALID`、`PTO_A1_TEST_STRIDED_DMA`、
`PTO_A1_TEST_ATOMIC_STORE` 和 `__PTO_AUTO__` 选择 CCEC 负向检查，须以对应的 A1 约束诊断拒绝编译。

FP32 用例使用 `tests/compile/a1_fp32_vector.cpp`，CPU/CCEC 编译命令同上。
默认形状包含 2048 个元素，大形状包含 16384 个元素（256 个 repeat、两个 64 KiB 输入缓冲）。
启动时传入对应元素数量和 `fp32`：

```bash
ccec -c -std=c++17 -O2 -x cce --cce-aicore-only --cce-aicore-arch=dav-m200 \
  -I include tests/compile/a1_fp32_vector.cpp -o /tmp/pto-a1-fp32-vector.o
ASCEND_RT_VISIBLE_DEVICES=4 /tmp/pto-a1-launch /tmp/pto-a1-fp32-vector.o 0 2048 fp32
```

Tensor 与 tile 的数据类型必须一致。FP16 设置两段 64-bit 向量 mask，FP32 清空高位 mask。
每次调用都会显式设置 mask，包括连续执行 FP16 与 FP32 内核的情况。launcher 默认仍使用 FP16。

FP32 的负向模式 `PTO_A1_TEST_MIXED_ADD` 校验操作数数据类型一致；
`PTO_A1_TEST_PARTIAL_REPEAT` 是单行 FP32 尾部的正向测试。

同一静态 FP16/FP32 配置也支持 `TSUB` 和 `TMUL`。CCEC 编译时可使用
`tests/compile/a1_tsub_vector.cpp` 或 `tests/compile/a1_tmul_vector.cpp`；参考
启动器分别传入 `2048 fp32 sub` 或 `2048 fp32 mul`。乘法检查保留零的符号位。
形状、数据类型、掩码、事件和重复次数限制与 `TADD` 相同；未验证的操作仍不开放。

A1 FP16/FP32 ND 访存支持大于 tile 宽度的静态行步长，每行须按 32 字节对齐。超过 4095 行的多 burst 传输分段执行，GM 间距超过 65535 个块时逐行拷贝。写回保留 tile 外的列。

A1 `TADD`、`TSUB`、`TMUL` 支持静态完整 tile 的最后一次向量重复使用掩码；DMA 行仍须按 32 字节对齐。尾部掩码排除相邻 tile，指令执行后恢复完整重复的掩码。

A1 提供限定窄矩阵配置：K=N=16，M 为 16 至 2048 之间按 16 对齐的静态完整行。对齐的 ND FP16 输入加载至 ZZ 或 NZ Mat，`TMOV` 准备 NZ Left 与 ZN Right，`TMATMUL` 初始化 FP32 NZ 累加器。显式 `TMOV<Vec, Acc, AccToVecMode::SingleModeVec0>` 将结果写至匹配的 FP32 ND Vec，再经 `TSTORE` 写回。ZZ/NZ Mat 和 NZ Acc 在仅有一个外层列时，其多行分形与 ND 保持同样元素顺序；该等价关系不适用于更大 K/N。部分有效行、融合 relu、双 worker 模式及 Acc 直接写回仍不支持。

`tests/compile/a1_single_fractal_gemm.cpp` 保留最初 16x16 参考及无效类型、布局、模式变体；`PTO_A1_TEST_MATRIX_NZ_INPUT` 选择标准 NZ Mat。`tests/compile/a1_narrow_gemm_rows.cpp` 以 `PTO_A1_TEST_ROWS` 选择 16、32、128、1024、2048 行 CPU/设备参考。上述源码配置须配合相应的前端和运行时版本，且完整行 tile 受各层存储预算约束。

A1 `TCVT` 支持静态完整行主序 ND Vec tile 的 FP16 与 FP32 互转，限定 `RoundMode::CAST_NONE` 及 `SaturationMode::OFF`。两种类型的行均须按 32 字节对齐。硬件 FP32 到 FP16 转换按最近偶数舍入，并保留有符号零、次正规数及无穷值；保留 NaN 位置但不保证其 payload。超过 255 次重复会拆分，最后不足一次重复的元素使用掩码。缓冲区须互不重叠，唯独允许 FP32 到 FP16 缩窄时起始地址完全相同。不支持的运行时模式或重叠即使未启用调试断言也会 trap。同类型转换、scratch 参数及其他类型互转仍未开放。`tests/compile/a1_floating_conversion.cpp` 覆盖 CPU/设备参考及扩宽、重复边界、尾部和缩窄复用变体。

`TMATMUL_ACC(dst, acc, lhs, rhs)` 支持上述窄矩阵配置的原位累加。先用 `TMATMUL` 初始化；`dst` 与 `acc` 必须为同类型且指向同一 L0C 存储，否则 release 构建也会 trap。每次累加 MAD 前执行 PIPE_M barrier，保证 L0C 写后读依赖。暂不支持 AccPhase 参数。`tests/compile/a1_narrow_gemm_accumulation.cpp` 验证连续三次 MAD 累加。

A1 Mat `TLOAD` 支持静态 Nx16 FP16 面板的跨步 ND 行，行跨度至少 16 且按 16 元素对齐。32 字节块间隔不超过 65535 时使用逐行 burst，否则分行拷贝以避免截断。目标仍是连续 ZZ/NZ 面板；测试包含非零对齐列偏移和大间隔。此路径可从更宽的输入矩阵中加载 K/N 面板，MAD 本身仍限定 K=N=16。

A1 `TEXP` 和 `TDIV` 支持完整静态 ND FP16/FP32 Vec tile，使用默认精度（default precision）。两者共用算术掩码和 255 次重复分批路径，支持输入输出地址完全相同的原位执行，不支持部分重叠。高精度变体会被拒绝。`tests/compile/a1_exp_div_vector.cpp` 覆盖两种指令和原位变体。指数和除法使用硬件近似运算；除零及极端数值遵循目标行为，不承诺跨平台 IEEE 语义。

A1 `TEXPANDS` 将 FP16/FP32 标量广播（scalar broadcast）到静态完整 ND Vec tile。它与逐元素运算共用尾部掩码及 255 次重复分批。`tests/compile/a1_scalar_broadcast.cpp` 和配套原生 pytest 覆盖标量逐位结果、带符号零和相邻缓冲区哨兵。

A1 `TROWSUM`/`TROWMAX` 将完整静态行主序 FP16/FP32 输入归约为紧凑列主序（packed column-major）Mx1 Vec 结果，需要不重叠的同类型完整大小行主序临时缓冲区。M 须使紧凑结果满足 32 字节对齐，普通输入行仍须对齐。连续 Mx1 的 ND/DN 传输物理等价，支持对齐的分区偏移。m200 行最大值的数值/索引对经显式向量/标量同步后打包为数值。`TCOLSUM`/`TCOLMAX` 沿行维归约为同列宽的 1xN 行主序结果；列求和目前仅支持无临时缓冲区的顺序形式。输入、输出和临时缓冲区不得重叠。求和保持 tile 数据类型，需要 FP32 累加时应使用 FP32 输入。原生回归使用有限数值，NaN 和带符号零的并列值行为取决于目标硬件。动态形状、未对齐尾部及归约结果的行列广播仍需独立支持。
