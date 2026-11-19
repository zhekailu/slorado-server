/*
 * slorado-server: a TCP basecall server for readfish adaptive sampling.
 *
 * readfish's NetworkCaller sends each decision-loop batch of raw read chunks
 * (see protocol.h); the server basecalls them with slorado's own pipeline and
 * replies with one FASTQ record per read. Only slorado's public batch API is
 * used, so src/ stays untouched and upstream slorado updates merge cleanly.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <getopt.h>
#include <time.h>
#include <string>
#include <vector>

#include "slorado.h"
#include "basecall.h"
#include "error.h"
#include "misc.h"
#include "socket.h"
#include "protocol.h"

#define DEFAULT_PORT 9500
#define DEFAULT_MAX_REQUEST_MIB 256
#define DEFAULT_MIN_SAMPLES 100
#define MAX_BEAM_WIDTH_OPT 32   /* openfish's MAX_BEAM_WIDTH; openfish rejects anything wider */

/* readfish sends "RF-<read id>" but looks results up by the bare read id */
#define READFISH_ID_PREFIX "RF-"

typedef struct {
    uint32_t max_request;   // max frame size in bytes
    uint32_t min_samples;   // reads shorter than this get an empty record
    FILE *timing;           // per-batch timing TSV, or NULL
    int beam_width;         // CRF beam search width; 0 keeps openfish's default
    int pipeline;           // CPU beam-search pipeline sub-batch size in chunks; 0 = off
    int pipeline_threads;   // its CPU threads; 0 = one less than the online CPUs
} server_opt_t;

/* server-side time spent on one logical batch, from receipt of fragment 0 to the reply being sent */
typedef struct {
    uint32_t batch;
    uint32_t fragments;
    uint32_t reads;
    uint32_t basecalled_reads;
    uint64_t samples;
    uint64_t bases;
    double recv;        // fragment 0 received -> last fragment received
    double unpack;      // payload -> slow5_rec_t
    double preprocess;
    double load;        // chunks -> runner input tensor, summed over runners
    double h2d;         // input tensor host -> device copy, summed over runners
    double inference;   // model forward pass, summed over runners
    double transpose;   // scores transpose to TNC before decode, summed over runners (TSV column reshape_ms)
    double decode;      // CRF beam-search decode of the scores, summed over runners
    double postprocess;
    double fastq;
    double send;
    double total;
} batch_timing_t;

static struct option long_options[] = {
    {"threads", required_argument, 0, 't'},          //0
    {"gpu-batchsize", required_argument, 0, 'C'},    //1
    {"chunk-size", required_argument, 0, 'c'},       //2
    {"overlap", required_argument, 0, 'p'},          //3
    {"device", required_argument, 0, 'x'},           //4
    {"verbose", required_argument, 0, 'v'},          //5
    {"help", no_argument, 0, 'h'},                   //6
    {"version", no_argument, 0, 'V'},                //7
    {"flash", required_argument, 0, 0},              //8
    {"max-request-mib", required_argument, 0, 0},    //9
    {"min-samples", required_argument, 0, 0},        //10
    {"timing", required_argument, 0, 0},             //11
    {"model", required_argument, 0, 0},              //12
    {"port", required_argument, 0, 0},               //13
    {"beam-width", required_argument, 0, 0},         //14
    {"pipeline", required_argument, 0, 0},           //15
    {"pipeline-threads", required_argument, 0, 0},   //16
    {0, 0, 0, 0}};

static void print_help_msg(FILE *fp_help, slorado_opt_t opt) {
    fprintf(fp_help, "usage: slorado-server --model FILE [--port INT] [options]\n");
    fprintf(fp_help, "required arguments:\n");
    fprintf(fp_help, "  --model FILE                the basecaller model to run.\n");
    fprintf(fp_help, "\nbasic options:\n");
    fprintf(fp_help, "  --port INT                  TCP port to listen on (matches readfish network-latency's default) [%d]\n", DEFAULT_PORT);
    fprintf(fp_help, "  -t INT                      number of processing threads [%d]\n", opt.num_thread);
    fprintf(fp_help, "  -C INT                      gpu batch size (max number of chunks loaded at once) [auto]\n");
    fprintf(fp_help, "  -c INT                      chunk size [model config or %d]\n", DEFAULT_CHUNK_SIZE);
    fprintf(fp_help, "  -p INT                      overlap [model config or %d]\n", DEFAULT_OVERLAP);
    fprintf(fp_help, "  -x DEVICE                   specify device [%s]\n", opt.device);
    fprintf(fp_help, "  -h                          shows help message and exits\n");
    fprintf(fp_help, "  --flash=yes|no              use flash attention for better performance [%s]\n", (opt.flag & SLORADO_FLASH) ? "yes" : "no");
    fprintf(fp_help, "  --verbose INT               verbosity level [%d]\n", (int)get_log_level());
    fprintf(fp_help, "  --version                   print version\n");
    fprintf(fp_help, "\nserver options:\n");
    fprintf(fp_help, "  --max-request-mib INT       largest request frame accepted [%d]\n", DEFAULT_MAX_REQUEST_MIB);
    fprintf(fp_help, "  --min-samples INT           reads shorter than this are returned with an empty sequence [%d]\n", DEFAULT_MIN_SAMPLES);
    fprintf(fp_help, "  --timing FILE               write per-batch server-side timings as TSV\n");
    fprintf(fp_help, "  --beam-width INT            CRF beam search width, 1-%d [openfish default, %zu]\n", MAX_BEAM_WIDTH_OPT, openfish_decoder_default_opts().beam_width);
    fprintf(fp_help, "  --pipeline INT              integrated GPUs (Jetson): split batches of >= 2*INT chunks into sub-batches of\n"
                     "                              INT to 2*INT-1 chunks and run beam search, qualities and sequence on the CPU,\n"
                     "                              overlapped with the GPU (250 suits an Orin Nano) [0 = off]\n");
    fprintf(fp_help, "  --pipeline-threads INT      CPU threads for --pipeline [online CPUs - 1]\n");
}

/* Per-batch timings use a monotonic clock: slorado's realtime() is gettimeofday(),
   which steps when the system clock is adjusted (NTP on the Jetson, host sync on WSL). */
static double monotime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* slorado's per-runner counters are cumulative; diffing them around slorado_process_db()
   gives this batch's share. Summed over runners, so with one device it is wall time. */
static runner_stat_t runner_times(const slorado_core_t *core) {
    runner_stat_t sum;
    memset(&sum, 0, sizeof(sum));
    for (const runner_stat_t *ts : *core->runner_stats) {
        sum.time_accept += ts->time_accept;
        sum.time_h2d_transfer += ts->time_h2d_transfer;
        sum.time_infer += ts->time_infer;
        sum.time_transpose += ts->time_transpose;
        sum.time_decode += ts->time_decode;
    }
    return sum;
}

static void append_fastq(std::string &out, const char *read_id, const std::string &seq, const std::string &qstr) {
    size_t prefix_len = strlen(READFISH_ID_PREFIX);
    if (strncmp(read_id, READFISH_ID_PREFIX, prefix_len) == 0) read_id += prefix_len;
    out.push_back('@');
    out.append(read_id);
    out.push_back('\n');
    out.append(seq);
    out.append("\n+\n", 3);
    out.append(qstr);
    out.push_back('\n');
}

/* Basecall recs and append one FASTQ record per read, in request order, to msg.
   Takes ownership of recs and frees them. */
static void basecall_batch(slorado_core_t *core, std::vector<slow5_rec_t *> &recs, const server_opt_t *sopt,
                           std::string &msg, batch_timing_t *t) {
    const int32_t n = (int32_t)recs.size();

    // too-short reads stay out of slorado: a near-empty chunk can make the decoder exit()
    std::vector<int32_t> db_index(n, -1);
    int32_t n_call = 0;
    for (int32_t i = 0; i < n; i++) {
        t->samples += recs[i]->len_raw_signal;
        if (recs[i]->len_raw_signal >= sopt->min_samples) db_index[i] = n_call++;
    }
    t->reads = n;
    t->basecalled_reads = n_call;

    slorado_db_t *db = NULL;
    if (n_call > 0) {
        core->opt.batch_size = n_call; // slorado_init_db sizes the batch from this
        db = slorado_init_db(core);
        for (int32_t i = 0; i < n; i++) {
            if (db_index[i] >= 0) db->slow5_rec[db_index[i]] = recs[i]; // db now owns it
        }
        db->n_rec = n_call;

        // preprocess -> basecall -> postprocess; records are already decoded so slorado skips parsing.
        // Stage times come from slorado's cumulative counters (realtime(), not monotime()).
        double pre0 = core->time_preproc, post0 = core->time_postproc;
        runner_stat_t s0 = runner_times(core);
        slorado_process_db(core, db);
        runner_stat_t s1 = runner_times(core);
        t->preprocess = core->time_preproc - pre0;
        t->load = s1.time_accept - s0.time_accept;
        t->h2d = s1.time_h2d_transfer - s0.time_h2d_transfer;
        t->inference = s1.time_infer - s0.time_infer;
        t->transpose = s1.time_transpose - s0.time_transpose;
        t->decode = s1.time_decode - s0.time_decode;
        t->postprocess = core->time_postproc - post0;
    }

    double e = monotime();
    static const std::string empty;
    for (int32_t i = 0; i < n; i++) {
        int32_t j = db_index[i];
        if (j >= 0) {
            append_fastq(msg, recs[i]->read_id, (*db->sequence)[j], (*db->qstring)[j]);
            t->bases += (*db->sequence)[j].size();
        } else {
            append_fastq(msg, recs[i]->read_id, empty, empty);
        }
    }
    t->fastq = monotime() - e;

    if (db != NULL) slorado_free_db(db);
    for (int32_t i = 0; i < n; i++) {
        if (db_index[i] < 0) slow5_rec_free(recs[i]);
    }
    recs.clear();
}

static int send_response(int fd, uint32_t batch, uint32_t status, std::string &msg) {
    nt_encode_response_header((unsigned char *)&msg[0], batch, status, (uint32_t)(msg.size() - NT_RESPONSE_HEADER));
    return send_full_msg(fd, msg.data(), (uint32_t)msg.size());
}

static int send_error(int fd, uint32_t batch, const char *error) {
    std::string msg(NT_RESPONSE_HEADER, '\0');
    msg.append(error);
    return send_response(fd, batch, NT_STATUS_ERROR, msg);
}

static void free_recs(std::vector<slow5_rec_t *> &recs) {
    for (slow5_rec_t *rec : recs) slow5_rec_free(rec);
    recs.clear();
}

static void write_timing(FILE *fp, const batch_timing_t *t) {
    fprintf(fp, "%u\t%u\t%u\t%u\t%lu\t%lu\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n",
            t->batch, t->fragments, t->reads, t->basecalled_reads, (unsigned long)t->samples, (unsigned long)t->bases,
            t->recv * 1e3, t->unpack * 1e3, t->preprocess * 1e3, t->load * 1e3, t->h2d * 1e3,
            t->inference * 1e3, t->transpose * 1e3, t->decode * 1e3,
            t->postprocess * 1e3,
            t->fastq * 1e3, t->send * 1e3, t->total * 1e3);
}

/* Serve one client until it disconnects. Returns 0 on an orderly disconnect,
   -1 when the connection has to be dropped. */
static int serve(int fd, slorado_core_t *core, const server_opt_t *sopt) {
    bool active = false;                 // a batch is being accumulated
    uint32_t active_batch = 0, active_fragments = 0, expected_fragment = 0;
    bool has_failed = false;             // remaining fragments of a rejected batch are dropped quietly
    uint32_t failed_batch = 0;
    std::vector<slow5_rec_t *> recs;
    double t_first = 0;                  // when fragment 0 of the active batch arrived
    batch_timing_t t;
    memset(&t, 0, sizeof(t));

    for (;;) {
        void *raw = NULL;
        uint32_t size = 0;
        int rc = recv_full_msg(fd, &raw, &size, sopt->max_request);
        if (rc != 0) {
            free_recs(recs);
            if (rc == 1) return 0;
            // after an oversized advertisement the stream position is lost, so the connection cannot continue
            if (rc == -2) {
                ERROR("request larger than --max-request-mib %u", sopt->max_request >> 20);
            } else {
                ERROR("receive failed: %s", strerror(errno));
            }
            return -1;
        }
        double t_recv = monotime();

        nt_request_t req;
        const char *error = nt_parse_request((const unsigned char *)raw, size, &req);

        if (!error && req.fragment == 0) {
            if (active) {
                WARNING("batch %u abandoned after %u of %u fragments", active_batch, expected_fragment, active_fragments);
                free_recs(recs);
            }
            active = true;
            active_batch = req.batch;
            active_fragments = req.fragment_count;
            expected_fragment = 0;
            has_failed = false;
            memset(&t, 0, sizeof(t));
            t.batch = req.batch;
            t.fragments = req.fragment_count;
        } else if (!error && has_failed && req.batch == failed_batch) {
            free(raw); // already answered with an error
            continue;
        }

        if (!error && (!active || req.batch != active_batch || req.fragment_count != active_fragments ||
                       req.fragment != expected_fragment)) {
            error = "unexpected batch fragment";
        }

        if (!error) {
            double a = monotime();
            error = nt_decode_reads(req.reads, req.reads_len, recs);
            t.unpack += monotime() - a;
        }
        free(raw);

        if (error) {
            ERROR("batch=%u fragment=%u/%u error=%s", req.batch, req.fragment + 1, req.fragment_count, error);
            free_recs(recs);
            active = false;
            has_failed = true;
            failed_batch = req.batch;
            // a bad batch gets status 1 but keeps the connection for the next batch
            if (send_error(fd, req.batch, error)) return -1;
            continue;
        }

        if (expected_fragment == 0) t_first = t_recv;
        expected_fragment++;
        if (expected_fragment < active_fragments) continue;
        t.recv = t_recv - t_first;

        std::string msg(NT_RESPONSE_HEADER, '\0');
        basecall_batch(core, recs, sopt, msg, &t);
        active = false;

        double a = monotime();
        rc = send_response(fd, active_batch, NT_STATUS_OK, msg);
        double b = monotime();
        t.send = b - a;
        t.total = b - t_first;
        if (rc) {
            ERROR("send failed: %s", strerror(errno));
            return -1;
        }

        if (sopt->timing != NULL) write_timing(sopt->timing, &t);
        if (get_log_level() >= LOG_DBUG) {
            fprintf(stderr, "[%s] batch %u: %u reads (%u basecalled), %lu samples, %lu bases, %.1f ms (inference %.1f ms, decode %.1f ms)\n",
                    __func__, t.batch, t.reads, t.basecalled_reads, (unsigned long)t.samples, (unsigned long)t.bases,
                    t.total * 1e3, t.inference * 1e3, t.decode * 1e3);
        }
    }
}

int main(int argc, char* argv[]) {
    double realtime0 = realtime();

    const char* optstring = "t:C:c:p:x:v:hV";
    int longindex = 0;
    int32_t c = -1;
    FILE *fp_help = stderr;

    slorado_opt_t opt;
    slorado_init_opt(&opt); // initialise options to defaults

    server_opt_t sopt;
    sopt.max_request = (uint32_t)DEFAULT_MAX_REQUEST_MIB << 20;
    sopt.min_samples = DEFAULT_MIN_SAMPLES;
    sopt.timing = NULL;
    sopt.beam_width = 0;
    sopt.pipeline = 0;
    sopt.pipeline_threads = 0;
    const char *timing_path = NULL;
    char *model = NULL;
    long port = DEFAULT_PORT;

    // parse the user args
    while ((c = getopt_long(argc, argv, optstring, long_options, &longindex)) >= 0) {
        if (c == 'C') {
            opt.gpu_batch_size = atoi(optarg);
            if (opt.gpu_batch_size < 1) {
                ERROR("Batch size should larger than 0. You entered %d", opt.gpu_batch_size);
                exit(EXIT_FAILURE);
            }
            opt.mod_gpu_batch_size = opt.gpu_batch_size;
        } else if (c == 't') {
            opt.num_thread = atoi(optarg);
            if (opt.num_thread < 1) {
                ERROR("Number of threads should larger than 0. You entered %d", opt.num_thread);
                exit(EXIT_FAILURE);
            }
        } else if (c == 'v') {
            int v = atoi(optarg);
            set_log_level((enum log_level_opt)v);
            set_openfish_log_level((enum openfish_log_level_opt)v);
        } else if (c == 'x') {
            opt.device = optarg;
        } else if (c == 'c') {
            opt.chunk_size = atoi(optarg);
            if (opt.chunk_size < 1) {
                ERROR("Chunk size should larger than 0. You entered %zu", opt.chunk_size);
                exit(EXIT_FAILURE);
            }
        } else if (c == 'p') {
            opt.overlap = atoi(optarg);
            if (opt.overlap < 1) {
                ERROR("Overlap should larger than 0. You entered %d", opt.overlap);
                exit(EXIT_FAILURE);
            }
        } else if (c == 'V') {
            fprintf(stdout, "slorado-server %s\n", SLORADO_VERSION);
            exit(EXIT_SUCCESS);
        } else if (c == 'h') {
            fp_help = stdout;
        } else if (c == 0 && longindex == 8) { // flash attention
            yes_or_no(&opt.flag, SLORADO_FLASH, long_options[longindex].name, optarg, 1);
        } else if (c == 0 && longindex == 9) { // max request size
            long mib = atol(optarg);
            if (mib < 1 || mib > 4095) {
                ERROR("--max-request-mib should be between 1 and 4095. You entered %ld", mib);
                exit(EXIT_FAILURE);
            }
            sopt.max_request = (uint32_t)mib << 20;
        } else if (c == 0 && longindex == 10) { // min samples
            long min_samples = atol(optarg);
            if (min_samples < 1) {
                ERROR("--min-samples should larger than 0. You entered %ld", min_samples);
                exit(EXIT_FAILURE);
            }
            sopt.min_samples = (uint32_t)min_samples;
        } else if (c == 0 && longindex == 11) { // timing tsv
            timing_path = optarg;
        } else if (c == 0 && longindex == 12) { // model
            model = optarg;
        } else if (c == 0 && longindex == 14) { // beam width
            char *end = NULL;
            long bw = strtol(optarg, &end, 10);
            if (*end != '\0' || bw < 1 || bw > MAX_BEAM_WIDTH_OPT) {
                ERROR("--beam-width should be between 1 and %d. You entered %s", MAX_BEAM_WIDTH_OPT, optarg);
                exit(EXIT_FAILURE);
            }
            sopt.beam_width = (int)bw;
        } else if (c == 0 && (longindex == 15 || longindex == 16)) { // pipeline, pipeline-threads
            char *end = NULL;
            long v = strtol(optarg, &end, 10);
            if (*end != '\0' || v < 0 || v > 1000000) {
                ERROR("--%s should be a non-negative integer. You entered %s", long_options[longindex].name, optarg);
                exit(EXIT_FAILURE);
            }
            if (longindex == 15) sopt.pipeline = (int)v;
            else sopt.pipeline_threads = (int)v;
        } else if (c == 0 && longindex == 13) { // port
            char *end = NULL;
            port = strtol(optarg, &end, 10);
            if (*end != '\0' || port < 1 || port > 65535) {
                ERROR("Invalid port: %s", optarg);
                exit(EXIT_FAILURE);
            }
        } else {
            print_help_msg(stderr, opt);
            exit(EXIT_FAILURE);
        }
    }

    size_t max_input_tensor_len = 10000 * 6000; // 10k chunk len, 6k gpu batch size, as in basecaller_main

    if (opt.chunk_size > 0 && (size_t)opt.chunk_size * opt.gpu_batch_size > max_input_tensor_len) {
        ERROR("Your input tensor size: %zu (chunk_size * gpu_batch_size) exceeds maximum allowed size: %zu", (size_t)opt.chunk_size * opt.gpu_batch_size, max_input_tensor_len);
        exit(EXIT_FAILURE);
    }

    if (opt.chunk_size > 0 && opt.overlap > 0 && (size_t)opt.overlap >= opt.chunk_size) {
        ERROR("Your overlap: %d should be lesser than chunk size: %ld", opt.overlap, opt.chunk_size);
        exit(EXIT_FAILURE);
    }

    if (fp_help == stdout) {
        print_help_msg(fp_help, opt);
        exit(EXIT_SUCCESS);
    }

    if (model == NULL || argc != optind) {
        if (model == NULL) {
            ERROR("%s", "--model is required");
        } else {
            ERROR("Unexpected argument: %s (model and port are given with --model and --port)", argv[optind]);
        }
        print_help_msg(stderr, opt);
        exit(EXIT_FAILURE);
    }

    if (timing_path != NULL) {
        sopt.timing = fopen(timing_path, "w");
        F_CHK(sopt.timing, timing_path);
        setvbuf(sopt.timing, NULL, _IOLBF, 0); // the server is usually stopped with Ctrl-C, so keep rows flushed
        fprintf(sopt.timing, "batch\tfragments\treads\tbasecalled_reads\tsamples\tbases\trecv_ms\tunpack_ms\t"
                             "preprocess_ms\tload_ms\tcopyToGPU_ms\tinference_ms\treshape_ms\tdecode_ms\tpostprocess_ms\tfastq_ms\tsend_ms\tserver_ms\n");
    }

    // load the model once; it is reused for every client and batch
    // no slow5 file: reads arrive over the socket, so batch_size is set per request
    slorado_core_t *core = slorado_init_core(opt, model, NULL);
    if (sopt.beam_width > 0) core->decoder_opts.beam_width = (size_t)sopt.beam_width;
    if (sopt.pipeline > 0) {
        int threads = sopt.pipeline_threads;
        if (threads <= 0) {
            long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
            threads = ncpu > 1 ? (int)ncpu - 1 : 1;
        }
        basecall_set_pipeline(sopt.pipeline, threads);
    }

    fprintf(stderr, "\nslorado-server version %s\n", SLORADO_VERSION);
    fprintf(stderr, "model path:         %s\n", model);
    fprintf(stderr, "port:               %ld\n", port);
    fprintf(stderr, "device:             %s\n", opt.device);
    fprintf(stderr, "chunk size:         %zu\n", core->chunk_size);
    fprintf(stderr, "gpu batch size:     %d\n", core->opt.gpu_batch_size);
    fprintf(stderr, "no. threads:        %d\n", opt.num_thread);
    fprintf(stderr, "overlap:            %d\n", core->opt.overlap);
    fprintf(stderr, "min samples:        %u\n", sopt.min_samples);
    fprintf(stderr, "beam width:         %zu\n", core->decoder_opts.beam_width);
    if (basecall_pipeline_chunks() > 0) {
        fprintf(stderr, "pipeline:           CPU beam search, sub-batches of %d+ chunks\n", basecall_pipeline_chunks());
    }
    fprintf(stderr, "timing output:      %s\n", timing_path == NULL ? "none" : timing_path);
    fprintf(stderr, "model loaded in %.3f s\n\n", realtime() - realtime0);

    int listenfd = TCP_server_init((int)port);

    // serve one client at a time; go back to accept() after a disconnect so readfish can reconnect
    for (;;) {
        int connectfd = TCP_server_accept_client(listenfd);
        if (connectfd < 0) {
            if (errno == EINTR) continue;
            perror("Cannot accept");
            break;
        }
        if (serve(connectfd, core, &sopt)) {
            fprintf(stderr, "connection ended with an error\n");
        }
        TCP_server_disconnect_client(connectfd);
    }

    TCP_server_shutdown(listenfd);
    if (sopt.timing != NULL) fclose(sopt.timing);
    slorado_free_core(core, opt);
    return EXIT_FAILURE;
}
