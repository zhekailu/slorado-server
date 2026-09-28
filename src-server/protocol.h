#ifndef SLORADO_SERVER_PROTOCOL_H
#define SLORADO_SERVER_PROTOCOL_H

#include <stdint.h>
#include <string>
#include <vector>
#include <slow5/slow5.h>

/*
 * Application protocol carried inside each socket.h frame. It is fixed by
 * readfish's NetworkCaller (readfish/plugins/network.py) and is the same one
 * network-timing-edge speaks, so readfish needs no changes to use this server.
 *
 * Request ("NTQ1")                     Response ("NTR1")
 * offset  bytes  field                  offset  bytes  field
 *   0       4    magic                    0       4    magic
 *   4       4    version (= 2)            4       4    version (= 2)
 *   8       4    batch ID                 8       4    matching batch ID
 *  12       4    fragment index          12       4    status (0 ok, 1 error)
 *  16       4    fragment count          16       4    body length
 *  20       4    dataset-name length     20      ...   FASTQ or error text
 *  24       4    reads payload length
 *  28      ...   name, then reads
 *
 * Reads payload: u32 read count, then for every read a 32-byte header
 *   u32 id length, u32 sample count, u32 read tag,
 *   f32 DAQ offset, f32 DAQ scaling (range/digitisation), f32 sampling rate,
 *   u64 start sample
 * followed by the UTF-8 read ID and sample-count little-endian int16 samples.
 *
 * All header integers and floats are network byte order; only the signal is
 * little-endian. A logical batch larger than 256k samples arrives as several
 * fragments (0..count-1) and gets a single response.
 */

#define NT_VERSION 2u
#define NT_REQUEST_HEADER 28u
#define NT_RESPONSE_HEADER 20u
#define NT_READ_HEADER 32u
#define NT_MAX_NAME_LEN 240u
#define NT_MAX_ID_LEN 1024u

#define NT_STATUS_OK 0u
#define NT_STATUS_ERROR 1u

/* fixed request header fields; name and reads point into the received message */
typedef struct {
    uint32_t batch;
    uint32_t fragment;
    uint32_t fragment_count;
    uint32_t name_len;
    uint32_t reads_len;
    const unsigned char *name;
    const unsigned char *reads;
} nt_request_t;

/* Validate a request frame and fill req. Returns NULL on success, otherwise a
   static error message. req->batch is filled whenever the frame is long enough
   to hold it, so malformed requests can still be answered with a matching ID. */
const char *nt_parse_request(const unsigned char *msg, uint32_t size, nt_request_t *req);

/* Decode a reads payload, appending one slow5_rec_t per read to recs. The
   records carry read_id, raw_signal, len_raw_signal, offset, and the scaling as
   range with digitisation = 1, which is all slorado's scale_signal() uses.
   Returns NULL on success, otherwise a static error message; records appended
   before an error stay in recs and remain the caller's to free. */
const char *nt_decode_reads(const unsigned char *p, uint32_t size, std::vector<slow5_rec_t *> &recs);

/* Write the 20-byte response header for a body of body_len bytes. */
void nt_encode_response_header(unsigned char *hdr, uint32_t batch, uint32_t status, uint32_t body_len);

/* Parse a response frame. Returns NULL on success and points body into msg. */
const char *nt_parse_response(const unsigned char *msg, uint32_t size, uint32_t *batch, uint32_t *status,
                              const unsigned char **body, uint32_t *body_len);

/* Client side: append one read to a reads payload (the leading u32 count is
   written by nt_encode_request). */
void nt_encode_read(std::string &reads, const char *read_id, uint32_t read_tag, float offset, float scaling,
                    float sampling_rate, uint64_t start_sample, const int16_t *signal, uint32_t n_samples);

/* Client side: build a complete request frame from a reads payload holding n_reads reads. */
std::string nt_encode_request(uint32_t batch, uint32_t fragment, uint32_t fragment_count, const std::string &name,
                              uint32_t n_reads, const std::string &reads);

#endif
