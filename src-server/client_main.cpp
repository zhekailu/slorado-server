/*
 * slorado-client: send a BLOW5 file to slorado-server the way readfish's
 * NetworkCaller does and print the returned FASTQ. For testing the server
 * without readfish.
 *
 * Like readfish's load_blow5/_package_reads, each record becomes one read with
 * ID "RF-<read id>", scaling = range/digitisation, offset, sampling rate and
 * start sample 0. Like _split_reads, a batch is split into messages of at most
 * 256,000 samples.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <string>
#include <vector>
#include "socket.h"
#include "protocol.h"
#include "slow5/slow5.h"

#define MAX_MESSAGE_SAMPLES 256000
#define MAX_RESPONSE_MIB 64

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(int argc, char *argv[]){

	//argument check check
	if(argc < 4 || argc > 5){
		fprintf(stderr, "Usage: %s <ip_address> <port> <reads.blow5> [n_batches=1]\n",argv[0]);
		fprintf(stderr, "Sends the file as one readfish batch n_batches times; the first FASTQ response goes to stdout\n");
		exit(EXIT_FAILURE);
	}
	const char *ip = argv[1];
	int port = atoi(argv[2]);
	int n_batches = argc > 4 ? atoi(argv[4]) : 1;
	if (port < 1 || port > 65535 || n_batches < 1) {
		fprintf(stderr, "Invalid port or batch count\n");
		exit(EXIT_FAILURE);
	}

	slow5_file_t *sp = slow5_open(argv[3], "r");
	if (sp == NULL) {
		fprintf(stderr, "Could not open %s\n", argv[3]);
		exit(EXIT_FAILURE);
	}

	//encode every read once, grouping them into fragments of at most MAX_MESSAGE_SAMPLES
	std::vector<std::string> fragment_reads(1);
	std::vector<uint32_t> fragment_counts(1, 0);
	uint64_t fragment_samples = 0;
	uint32_t n_reads = 0;
	slow5_rec_t *rec = NULL;
	int ret;
	while ((ret = slow5_get_next(&rec, sp)) >= 0) {
		if (fragment_counts.back() > 0 && fragment_samples + rec->len_raw_signal > MAX_MESSAGE_SAMPLES) {
			fragment_reads.emplace_back();
			fragment_counts.push_back(0);
			fragment_samples = 0;
		}
		std::string read_id = std::string("RF-") + rec->read_id;
		nt_encode_read(fragment_reads.back(), read_id.c_str(), n_reads, (float)rec->offset,
		               (float)(rec->range / rec->digitisation), (float)rec->sampling_rate, 0,
		               rec->raw_signal, (uint32_t)rec->len_raw_signal);
		fragment_counts.back()++;
		fragment_samples += rec->len_raw_signal;
		n_reads++;
	}
	if (ret != SLOW5_ERR_EOF) {
		fprintf(stderr, "Error reading %s\n", argv[3]);
		exit(EXIT_FAILURE);
	}
	slow5_rec_free(rec);
	slow5_close(sp);
	if (n_reads == 0) {
		fprintf(stderr, "No reads in %s\n", argv[3]);
		exit(EXIT_FAILURE);
	}

	int socketfd = TCP_client_connect(ip, port);
	std::string name = "slorado-client";
	uint32_t n_fragments = (uint32_t)fragment_reads.size();

	for (int b = 1; b <= n_batches; b++) {
		double start = now_ms();
		for (uint32_t f = 0; f < n_fragments; f++) {
			std::string msg = nt_encode_request((uint32_t)b, f, n_fragments, name, fragment_counts[f], fragment_reads[f]);
			if (send_full_msg(socketfd, msg.data(), (uint32_t)msg.size())) {
				perror("Could not send request");
				exit(EXIT_FAILURE);
			}
		}

		void *raw = NULL;
		uint32_t size = 0;
		ret = recv_full_msg(socketfd, &raw, &size, (uint32_t)MAX_RESPONSE_MIB << 20);
		if (ret != 0) {
			fprintf(stderr, "Could not receive the response (%d)\n", ret);
			exit(EXIT_FAILURE);
		}
		double rtt = now_ms() - start;

		uint32_t batch, status, body_len;
		const unsigned char *body;
		const char *error = nt_parse_response((const unsigned char *)raw, size, &batch, &status, &body, &body_len);
		if (error == NULL && batch != (uint32_t)b) error = "response batch ID mismatch";
		if (error) {
			fprintf(stderr, "Invalid response: %s\n", error);
			exit(EXIT_FAILURE);
		}
		if (status != NT_STATUS_OK) {
			fprintf(stderr, "Server error for batch %d: %.*s\n", b, (int)body_len, (const char *)body);
			exit(EXIT_FAILURE);
		}

		fprintf(stderr, "batch %d: %u reads in %u fragment(s), %u FASTQ bytes, rtt %.3f ms\n",
		        b, n_reads, n_fragments, body_len, rtt);
		if (b == 1) fwrite(body, 1, body_len, stdout);
		free(raw);
	}

    //close the connection
	TCP_client_disconnect(socketfd);

	return 0;
}
