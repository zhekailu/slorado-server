# slorado-server

A TCP basecall server for readfish adaptive sampling. It speaks the protocol of
readfish-timing's `NetworkCaller` (`readfish network-latency`), so readfish
needs no changes. It basecalls every batch with slorado's own pipeline and
returns one FASTQ record per read.

All server code lives in `src-server/`. It calls slorado's public batch API
(`init_core`, `init_db`, `work_db`, `preprocess_signal_db`, `basecall_db`,
`postprocess_signal`, `free_db`), and nothing under `src/` is modified. That
lets upstream slorado updates merge without conflicts.

## Build

```bash
make cuda=1 slorado-server     # Jetson / CUDA
make slorado-server            # CPU only
make slorado-client            # optional test client
```

On the Jetson, `thirdparty/torch/libtorch` must be a Jetson (aarch64 + CUDA)
libtorch.

## Run

```bash
./slorado-server --model MODEL [--port PORT] [options]
./slorado-server -x cuda:0 -t 6 --timing server_timing.tsv \
  --model models/dna_r10.4.1_e8.2_400bps_fast@v5.2.0 --port 9500
```

`--model` is required. `--port` defaults to 9500, which is readfish
`network-latency`'s default `--port`.

The basecaller options are the same as `slorado basecaller`: `-t`, `-C`, `-c`,
`-p`, `-x`, `--flash`, and `--verbose`. Besides `--model` and `--port`, the
server adds three of its own:

| option | default | meaning |
|---|---|---|
| `--timing FILE` | off | Per-batch server-side TSV: recv, decode, preprocess, basecall, postprocess, fastq, send, and total ms. |
| `--min-samples INT` | 100 | Reads shorter than this are not basecalled and come back with an empty sequence. |
| `--max-request-mib INT` | 256 | Largest request frame accepted. |

The model is loaded once. The server handles one client at a time and accepts a
new connection after a disconnect. Stop it with Ctrl-C.

Point readfish at the server:

```bash
readfish network-latency --edge-host <server-ip> --port 9500 --data-dir ./data \
  --toml <TOML> --iterations 1000 --throttle 0.8 --output-dir results --experiment-name jetson_1
```

Test without readfish. This sends the file as readfish would and prints the
FASTQ; the optional last argument is the number of repeats.

```bash
./slorado-client 127.0.0.1 9500 reads.blow5 [n_batches] > out.fastq
```

## Protocol

Framing: a `uint32` big-endian length, then the payload. The payload formats
are documented in `protocol.h`.

1. **Batches.** One logical batch arrives as one or more `NTQ1` fragments of
   raw int16 signal, each at most 256k samples. The server waits for the last
   fragment, then replies once with `NTR1`.
2. **Read IDs.** readfish prefixes IDs with `RF-`. The server strips the prefix,
   because readfish looks results up by the bare ID.
3. **Calibration.** readfish sends `scaling = range/digitisation` and `offset`.
   The server stores these as `range = scaling`, `digitisation = 1`, and
   `offset`, which are the only values `scale_signal()` uses. Output is
   byte-identical to `slorado basecaller` on the same BLOW5.
4. **Errors.** A malformed request gets `status = 1` with the error text, and the
   connection stays open. Remaining fragments of that batch are dropped
   silently.
