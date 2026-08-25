# RDMA Async ST (HNS1825 Target Platform)

This directory contains the shared RDMA async ST implementation. `tput_async_rdma` validates remote WRITE,
`tget_async_rdma` validates remote READ, and `tput_async_notify_rdma` validates remote WRITE with a `Set`
notification. The three targets reuse the same RDMA test kernel implementation.

## Prerequisites

- An A5 environment with HNS1825 NICs and the matching driver/HCOMM stack
- MPI and HCCL configured as described in the top-level [tests README](../../../../../../README.md)
- Reachable RDMA NIC IPv4 addresses for all participating ranks

## Build and Run

Select the RDMA implementation before CMake configuration, then run the required RDMA ST target:

```bash
export PTO_RDMA_BACKEND=HNS_1825
python3 tests/script/run_st.py -r npu -v a5 -t comm/tput_async_rdma -d -n 2
python3 tests/script/run_st.py -r npu -v a5 -t comm/tget_async_rdma -d -n 2
python3 tests/script/run_st.py -r npu -v a5 -t comm/tput_async_notify_rdma -d -n 2
```

For the first `TPUT_ASYNC_NOTIFY` validation, run only the focused two-rank case:

```bash
export PTO_RDMA_BACKEND=HNS_1825
python3 tests/script/run_st.py -r npu -v a5 -t comm/tput_async_notify_rdma \
    -g TPutAsyncNotifyRdma.Int32SetAndCanaries -d -n 2
```

The case checks the remote payload, remote signal, and canaries on both sides of the signal, and waits for the returned
`AsyncEvent`. After observing the signal, the receiver maintains its data cache before checking the payload. The current
RDMA backend does not support `AtomicAdd`, so there is no corresponding RDMA case.

`PTO_RDMA_BACKEND` is read only while CMake configures this ST build. Unset, empty, or unsupported values build the tests without RDMA support. `run_st.py` rebuilds by default; after changing this variable, do not use `-w/--without-build` to reuse an existing binary.

## Endpoint Discovery

The bootstrap resolves each rank's physical device id and RDMA IPv4, then exchanges endpoint and registered-memory information through MPI. The local IPv4 lookup order is:

1. RoCE IPv4 in fixed `/etc/hccl_rootinfo.json` for the physical device.
2. HCOMM topology parsing of fixed `/var/run/ascend-topologyd/virtualTopology.xml`.
3. The test-only IP variables below.

The ST does not generate or modify either topology file and does not provide path overrides.

| Variable | Description |
|---|---|
| `PTO_RDMA_BACKEND` | Configure-time selector; the only supported value is `HNS_1825`. |
| `PTO_ROCE_PHYIDS` | Optional comma-separated physical device ids indexed by MPI rank. |
| `PTO_ROCE_LOCAL_IP` | Final fallback IPv4 for the current MPI process; set it separately for each rank when needed. |
| `PTO_ROCE_IPS` | Final fallback list containing exactly one IPv4 per MPI rank, in rank order. |
| `PTO_ROCE_BASE_PORT` | Common channel base port; default `60032`. |
| `PTO_ROCE_VERBOSE` | Set to `1` for endpoint, MR, channel, and cleanup progress logs. |
| `HCCL_RDMA_TC` | HCOMM traffic class; default `132`. |
| `HCCL_RDMA_SL` | HCOMM service level; default `4`. |

`PTO_ROCE_LOCAL_IP` takes precedence over `PTO_ROCE_IPS`. Both are ignored when root-info or virtual-topology lookup succeeds. All ranks must use the same base port and, when `PTO_ROCE_IPS` is used, the same rank-ordered list.

## Troubleshooting

- If CMake reports that RDMA is disabled, set `PTO_RDMA_BACKEND=HNS_1825` and reconfigure without `-w`.
- If endpoint discovery fails, verify the physical device mapping and the RoCE IPv4 in root-info or virtual topology, then use the test-only IP fallback if required.
- If HCOMM cannot load the HNS1825 verbs provider from its default search path, set `IBV_EXTEND_DRIVERS` to the driver-provided `libhrn5-rdmav34.so`.
- Set `PTO_ROCE_VERBOSE=1` to distinguish endpoint discovery, MR registration, channel setup, and cleanup failures.
