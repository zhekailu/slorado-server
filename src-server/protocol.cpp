#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include "protocol.h"

/* Use memcpy in the field helpers rather than unaligned integer casts, which
 * would be undefined or fault on some edge-device architectures. */
static uint32_t get32(const unsigned char *p) { uint32_t x; memcpy(&x, p, 4); return ntohl(x); }
static uint64_t get64(const unsigned char *p) { return ((uint64_t)get32(p) << 32) | get32(p + 4); }
static float getf32(const unsigned char *p) { uint32_t x = get32(p); float f; memcpy(&f, &x, 4); return f; }

static void put32(unsigned char *p, uint32_t x) { x = htonl(x); memcpy(p, &x, 4); }
static void append32(std::string &s, uint32_t x) { unsigned char b[4]; put32(b, x); s.append((const char *)b, 4); }
static void appendf32(std::string &s, float f) { uint32_t x; memcpy(&x, &f, 4); append32(s, x); }
static void append64(std::string &s, uint64_t x) { append32(s, (uint32_t)(x >> 32)); append32(s, (uint32_t)x); }

/* The signal is little-endian int16 on the wire (numpy "<i2"). */
static void copy_le16(int16_t *dst, const unsigned char *src, uint64_t n) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    for (uint64_t i = 0; i < n; i++) {
        dst[i] = (int16_t)((uint16_t)src[2 * i] | ((uint16_t)src[2 * i + 1] << 8));
    }
#else
    memcpy(dst, src, n * sizeof(int16_t));
#endif
}

const char *nt_parse_request(const unsigned char *msg, uint32_t size, nt_request_t *req) {
    memset(req, 0, sizeof(*req));
    if (size >= 12) req->batch = get32(msg + 8);

    /* Never access variable-length fields until the fixed header is known to
     * exist and its magic/version identify the expected protocol. */
    if (size < NT_REQUEST_HEADER || memcmp(msg, "NTQ1", 4) || get32(msg + 4) != NT_VERSION) {
        return "invalid request header";
    }
    req->fragment = get32(msg + 12);
    req->fragment_count = get32(msg + 16);
    req->name_len = get32(msg + 20);
    req->reads_len = get32(msg + 24);

    /* Exact equality rejects truncation and trailing bytes. */
    if ((uint64_t)NT_REQUEST_HEADER + req->name_len + req->reads_len != size) {
        return "reads byte count does not match request header";
    }
    if (req->fragment_count == 0 || req->fragment >= req->fragment_count) {
        return "invalid fragment index/count";
    }
    if (req->name_len == 0 || req->name_len > NT_MAX_NAME_LEN) {
        return "invalid dataset name length";
    }
    req->name = msg + NT_REQUEST_HEADER;
    req->reads = req->name + req->name_len;
    return NULL;
}

const char *nt_decode_reads(const unsigned char *p, uint32_t size, std::vector<slow5_rec_t *> &recs) {
    if (size < 4) return "reads payload too short";
    uint32_t count = get32(p);
    uint64_t offset = 4;

    for (uint32_t i = 0; i < count; i++) {
        if (offset + NT_READ_HEADER > size) return "truncated read header";
        const unsigned char *h = p + offset;
        uint32_t id_len = get32(h);
        uint32_t n_samples = get32(h + 4);
        float daq_offset = getf32(h + 12);
        float daq_scaling = getf32(h + 16);
        float sampling_rate = getf32(h + 20);
        offset += NT_READ_HEADER;

        if (id_len == 0 || id_len > NT_MAX_ID_LEN) return "invalid read ID length";
        if (offset + id_len + (uint64_t)n_samples * 2u > size) return "truncated read body";

        /* The ID becomes a FASTQ header token, so whitespace or control
         * characters would corrupt the response. */
        const unsigned char *id = p + offset;
        for (uint32_t j = 0; j < id_len; j++) {
            if (id[j] <= ' ' || id[j] == 0x7f) return "read ID contains whitespace or control characters";
        }

        slow5_rec_t *rec = (slow5_rec_t *)calloc(1, sizeof(slow5_rec_t));
        if (rec == NULL) return "out of memory";
        rec->read_id = (char *)malloc(id_len + 1);
        rec->raw_signal = (int16_t *)malloc(n_samples ? (size_t)n_samples * sizeof(int16_t) : 1);
        if (rec->read_id == NULL || rec->raw_signal == NULL) {
            slow5_rec_free(rec);
            return "out of memory";
        }
        memcpy(rec->read_id, id, id_len);
        rec->read_id[id_len] = '\0';
        rec->read_id_len = (slow5_rid_len_t)id_len;
        offset += id_len;

        copy_le16(rec->raw_signal, p + offset, n_samples);
        rec->len_raw_signal = n_samples;
        offset += (uint64_t)n_samples * 2u;

        /* scale_signal() only uses range/digitisation and offset, so the
         * readfish scaling (already range/digitisation) goes in range. */
        rec->digitisation = 1.0;
        rec->range = daq_scaling;
        rec->offset = daq_offset;
        rec->sampling_rate = sampling_rate;

        recs.push_back(rec);
    }

    return offset == size ? NULL : "trailing bytes after the last read";
}

void nt_encode_response_header(unsigned char *hdr, uint32_t batch, uint32_t status, uint32_t body_len) {
    memcpy(hdr, "NTR1", 4);
    put32(hdr + 4, NT_VERSION);
    put32(hdr + 8, batch);
    put32(hdr + 12, status);
    put32(hdr + 16, body_len);
}

const char *nt_parse_response(const unsigned char *msg, uint32_t size, uint32_t *batch, uint32_t *status,
                              const unsigned char **body, uint32_t *body_len) {
    if (size < NT_RESPONSE_HEADER || memcmp(msg, "NTR1", 4) || get32(msg + 4) != NT_VERSION) {
        return "invalid response header";
    }
    *batch = get32(msg + 8);
    *status = get32(msg + 12);
    *body_len = get32(msg + 16);
    if ((uint64_t)NT_RESPONSE_HEADER + *body_len != size) return "response body length mismatch";
    *body = msg + NT_RESPONSE_HEADER;
    return NULL;
}

void nt_encode_read(std::string &reads, const char *read_id, uint32_t read_tag, float offset, float scaling,
                    float sampling_rate, uint64_t start_sample, const int16_t *signal, uint32_t n_samples) {
    uint32_t id_len = (uint32_t)strlen(read_id);
    append32(reads, id_len);
    append32(reads, n_samples);
    append32(reads, read_tag);
    appendf32(reads, offset);
    appendf32(reads, scaling);
    appendf32(reads, sampling_rate);
    append64(reads, start_sample);
    reads.append(read_id, id_len);
    for (uint32_t i = 0; i < n_samples; i++) {
        uint16_t s = (uint16_t)signal[i];
        reads.push_back((char)(s & 0xff));
        reads.push_back((char)(s >> 8));
    }
}

std::string nt_encode_request(uint32_t batch, uint32_t fragment, uint32_t fragment_count, const std::string &name,
                              uint32_t n_reads, const std::string &reads) {
    std::string msg;
    msg.reserve(NT_REQUEST_HEADER + name.size() + 4 + reads.size());
    msg.append("NTQ1", 4);
    append32(msg, NT_VERSION);
    append32(msg, batch);
    append32(msg, fragment);
    append32(msg, fragment_count);
    append32(msg, (uint32_t)name.size());
    append32(msg, (uint32_t)(4 + reads.size()));
    msg.append(name);
    append32(msg, n_reads);
    msg.append(reads);
    return msg;
}
