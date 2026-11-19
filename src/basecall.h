/* @file basecall.h
**
** methods for base calling step
** @@
******************************************************************************/

#ifndef BASECALL
#define BASECALL

#include "slorado.h"

void basecall_db(core_t* core, db_t* db);
void mod_basecall_db(core_t* core, db_t* db);

/* CPU beam-search pipeline: split each batch of at least 2 * sub_chunks chunks into equal sub-batches of
   sub_chunks to 2 * sub_chunks - 1 chunks (0 = off, the default; smaller batches stay on the GPU) and run the beam search, qualities and sequence on cpu_threads CPU threads while the
   GPU runs the model and scans of the next sub-batch. Integrated GPUs only (disabled otherwise). */
void basecall_set_pipeline(int sub_chunks, int cpu_threads);
int basecall_pipeline_chunks(void);

void basecall_chunks(const core_t* core, const int runner_idx, const std::vector<basecall_chunk_t *> &chunks);
void mod_basecall_chunks(const core_t* core, const int runner_idx, const std::vector<mod_chunk_t *> &chunks);

#endif