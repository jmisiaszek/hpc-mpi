// ilu.cpp
#include "ilu.h"
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cassert>
#include <cmath>
#include <map>
#include <set>
#include <numeric>
#include <unordered_map>

struct CSRMatrix {
    int nrows;
    int ncols;
    std::vector<int> rowptr;
    std::vector<int> colidx;
    std::vector<double> vals;
};

struct ILUFact {
    MPI_Comm comm;
    int rank;
    int world_size;

    int N;
    int local_n;
    int row_offset;

    std::vector<int> row_offsets;

    int n_interior;
    int n_separator;

    // Local permutation: new_to_old[i] = old local index of new i-th row.
    std::vector<int> new_to_old;
    std::vector<int> old_to_new;

    // glob_old_to_new[g] = permuted global index of original global row/column
    // g (gathered so every rank can translate any column into it).
    std::vector<int> glob_old_to_new;

    CSRMatrix LU;

    // Remote rows received during factorization (indexed by global row id).
    struct RemoteRow { int diag_col; std::vector<int> cols; std::vector<double> vals; };
    std::unordered_map<int, RemoteRow> remote_rows;

    std::set<int> needed_global_rows;   // global row indices we depend on

    // We store the set of ranks we send separator rows to.
    std::set<int> send_to_ranks;    // ranks that depend on our rows
    std::set<int> recv_from_ranks;  // ranks we recv rows from (= lower ranks with our deps)

    // Cached request lists (built once, reused every iteration)
    std::vector<std::vector<int>> _send_requests;          // [rank] -> our global rows that rank needs
    std::map<int, std::vector<int>> _recv_requests;        // rank -> global rows we need from rank

    // Cached comm pattern for backward substitution / multiplication by U.
    std::map<int, std::vector<int>> bwd_need;   // higher rank -> cols we need from it
    std::map<int, std::vector<int>> bwd_serve;  // lower rank  -> our cols it needs
};

// ---------- Helpers ----------

static int owner_of(int g, int N, int world_size) {
    for (int r = world_size - 1; r >= 0; --r) {
        long long off = (long long)r * N / world_size;
        if (off <= g) {
            return r;
        }
    }
    return 0;
}

static int row_offset_of(int r, int N, int world_size) {
    int offset = (int)((long long)r * N / world_size);
    return offset;
}

// ---------- Step 1: distribute ----------
static void distribute_matrix(int N, int nnz,
                              const int* row, const int* col, const double* val,
                              ILUFact* ilu, CSRMatrix& localA)
{
    int rank = ilu->rank;
    int ws = ilu->world_size;

    ilu->row_offsets.resize(ws + 1);
    for (int r = 0; r <= ws; ++r)
        ilu->row_offsets[r] = row_offset_of(r, N, ws);

    ilu->N = N;
    ilu->row_offset = ilu->row_offsets[rank];
    ilu->local_n = ilu->row_offsets[rank + 1] - ilu->row_offsets[rank];

    if (rank == 0) {
        std::vector<std::vector<int>>    buf_row(ws), buf_col(ws);
        std::vector<std::vector<double>> buf_val(ws);

        // Count distribution
        std::vector<int> cnt_per_rank(ws, 0);
        for (int k = 0; k < nnz; ++k) {
            int r = owner_of(row[k], N, ws);
            buf_row[r].push_back(row[k]);
            buf_col[r].push_back(col[k]);
            buf_val[r].push_back(val[k]);
            cnt_per_rank[r]++;
        }

        for (int r = 1; r < ws; ++r) {
            int cnt = (int)buf_row[r].size();
            MPI_Send(&cnt, 1, MPI_INT, r, 100, ilu->comm);
            if (cnt > 0) {
                MPI_Send(buf_row[r].data(), cnt, MPI_INT,    r, 101, ilu->comm);
                MPI_Send(buf_col[r].data(), cnt, MPI_INT,    r, 102, ilu->comm);
                MPI_Send(buf_val[r].data(), cnt, MPI_DOUBLE, r, 103, ilu->comm);
            }
        }

        localA.nrows = ilu->local_n;
        localA.ncols = N;
        localA.rowptr.assign(localA.nrows + 1, 0);

        for (size_t k = 0; k < buf_row[0].size(); ++k) {
            int lr = buf_row[0][k] - ilu->row_offset;
            if (lr >= 0 && lr < localA.nrows) {
                localA.rowptr[lr + 1]++;
            }
        }
        for (int i = 0; i < localA.nrows; ++i)
            localA.rowptr[i + 1] += localA.rowptr[i];

        int total = localA.rowptr[localA.nrows];
        localA.colidx.assign(total, 0);
        localA.vals.assign(total, 0.0);

        std::vector<int> cursor(localA.nrows, 0);
        for (size_t k = 0; k < buf_row[0].size(); ++k) {
            int lr = buf_row[0][k] - ilu->row_offset;
            int pos = localA.rowptr[lr] + cursor[lr]++;
            localA.colidx[pos] = buf_col[0][k];
            localA.vals[pos]   = buf_val[0][k];
        }
    } else {
        int cnt = 0;
        MPI_Recv(&cnt, 1, MPI_INT, 0, 100, ilu->comm, MPI_STATUS_IGNORE);

        std::vector<int> rcv_row(cnt), rcv_col(cnt);
        std::vector<double> rcv_val(cnt);
        if (cnt > 0) {
            MPI_Recv(rcv_row.data(), cnt, MPI_INT,    0, 101, ilu->comm, MPI_STATUS_IGNORE);
            MPI_Recv(rcv_col.data(), cnt, MPI_INT,    0, 102, ilu->comm, MPI_STATUS_IGNORE);
            MPI_Recv(rcv_val.data(), cnt, MPI_DOUBLE, 0, 103, ilu->comm, MPI_STATUS_IGNORE);
        }

        localA.nrows = ilu->local_n;
        localA.ncols = N;
        localA.rowptr.assign(localA.nrows + 1, 0);

        for (int k = 0; k < cnt; ++k) {
            int lr = rcv_row[k] - ilu->row_offset;
            if (lr >= 0 && lr < localA.nrows) {
                localA.rowptr[lr + 1]++;
            }
        }
        for (int i = 0; i < localA.nrows; ++i)
            localA.rowptr[i + 1] += localA.rowptr[i];

        int total = localA.rowptr[localA.nrows];
        localA.colidx.assign(total, 0);
        localA.vals.assign(total, 0.0);

        std::vector<int> cursor(localA.nrows, 0);
        for (int k = 0; k < cnt; ++k) {
            int lr = rcv_row[k] - ilu->row_offset;
            int pos = localA.rowptr[lr] + cursor[lr]++;
            localA.colidx[pos] = rcv_col[k];
            localA.vals[pos]   = rcv_val[k];
        }
    }

    // Sort entries within each row
    for (int i = 0; i < localA.nrows; ++i) {
        int s = localA.rowptr[i], e = localA.rowptr[i + 1];
        std::vector<std::pair<int,double>> tmp(e - s);
        for (int k = s; k < e; ++k) tmp[k - s] = {localA.colidx[k], localA.vals[k]};
        std::sort(tmp.begin(), tmp.end(),
                  [](const auto& a, const auto& b){ return a.first < b.first; });
        for (int k = s; k < e; ++k) {
            localA.colidx[k] = tmp[k - s].first;
            localA.vals[k]   = tmp[k - s].second;
        }
    }
}

// ---------- Step 2: classify ----------
static void classify_rows(const CSRMatrix& A, ILUFact* ilu,
                          std::vector<char>& is_separator)
{
    int my_lo = ilu->row_offset;
    is_separator.assign(A.nrows, 0);

    int sep_count = 0;
    int interior_count = 0;

    for (int i = 0; i < A.nrows; ++i) {
        int has_lower = 0;
        for (int k = A.rowptr[i]; k < A.rowptr[i + 1]; ++k) {
            int c = A.colidx[k];
            if (c < my_lo) {
                is_separator[i] = 1;
                sep_count++;
                has_lower = 1;
                break;
            }
        }
        if (!has_lower) interior_count++;
    }
}

// ---------- Step 3: build permutation ----------
static void build_permutation(const std::vector<char>& is_sep, ILUFact* ilu)
{
    int n = (int)is_sep.size();

    ilu->new_to_old.clear();
    ilu->new_to_old.reserve(n);

    for (int i = 0; i < n; ++i)
        if (!is_sep[i]) ilu->new_to_old.push_back(i);
    ilu->n_interior = (int)ilu->new_to_old.size();

    for (int i = 0; i < n; ++i)
        if (is_sep[i]) ilu->new_to_old.push_back(i);
    ilu->n_separator = n - ilu->n_interior;

    ilu->old_to_new.assign(n, -1);
    for (int newi = 0; newi < n; ++newi) {
        ilu->old_to_new[ilu->new_to_old[newi]] = newi;
    }
}

// Gather all local permutations into one global old->new mapping, so that
// every rank can translate any column index into the permuted numbering.
static void build_global_permutation(ILUFact* ilu)
{
    int ws = ilu->world_size;
    std::vector<int> counts(ws), displs(ws);
    for (int r = 0; r < ws; ++r) {
        counts[r] = ilu->row_offsets[r + 1] - ilu->row_offsets[r];
        displs[r] = ilu->row_offsets[r];
    }

    std::vector<int> local_perm(ilu->local_n);
    for (int i = 0; i < ilu->local_n; ++i)
        local_perm[i] = ilu->row_offset + ilu->old_to_new[i];

    ilu->glob_old_to_new.assign(ilu->N, 0);
    MPI_Allgatherv(local_perm.data(), ilu->local_n, MPI_INT,
                   ilu->glob_old_to_new.data(), counts.data(), displs.data(),
                   MPI_INT, ilu->comm);
}

// ---------- Permute matrix (rows, and ALL columns through the global permutation) ----------
static void permute_matrix(const CSRMatrix& A, CSRMatrix& P, const ILUFact* ilu)
{
    int n = A.nrows;

    P.nrows = n;
    P.ncols = A.ncols;
    P.rowptr.assign(n + 1, 0);

    for (int newi = 0; newi < n; ++newi) {
        int oldi = ilu->new_to_old[newi];
        int nnz_i = A.rowptr[oldi + 1] - A.rowptr[oldi];
        P.rowptr[newi + 1] = nnz_i;
    }
    for (int i = 0; i < n; ++i) P.rowptr[i + 1] += P.rowptr[i];

    int total = P.rowptr[n];

    P.colidx.assign(total, 0);
    P.vals.assign(total, 0.0);

    for (int newi = 0; newi < n; ++newi) {
        int oldi = ilu->new_to_old[newi];
        int dst = P.rowptr[newi];
        int col_count = 0;

        for (int k = A.rowptr[oldi]; k < A.rowptr[oldi + 1]; ++k) {
            // Remap every column (local and remote) into the global permuted
            // numbering, so that all ranks talk about the same indices.
            P.colidx[dst] = ilu->glob_old_to_new[A.colidx[k]];
            P.vals[dst]   = A.vals[k];
            col_count++;
            ++dst;
        }

        int s = P.rowptr[newi], e = P.rowptr[newi + 1];
        std::vector<std::pair<int,double>> tmp(e - s);
        for (int k = s; k < e; ++k) tmp[k - s] = {P.colidx[k], P.vals[k]};
        std::sort(tmp.begin(), tmp.end(),
                  [](const auto& a, const auto& b){ return a.first < b.first; });
        for (int k = s; k < e; ++k) {
            P.colidx[k] = tmp[k - s].first;
            P.vals[k]   = tmp[k - s].second;
        }
    }
}

// ---------- Step 4: factorize interior rows ----------
static void ILU_factorize_interior(ILUFact* ilu)
{
    int my_lo = ilu->row_offset;
    CSRMatrix& U = ilu->LU;

    std::vector<int> diag_ptrs(ilu->n_interior, -1);
    for (int i = 0; i < ilu->n_interior; ++i) {
        int global_i = my_lo + i;
        for (int p = U.rowptr[i]; p < U.rowptr[i + 1]; ++p) {
            if (U.colidx[p] == global_i) {
                diag_ptrs[i] = p;
                break;
            }
        }
    }

    int elimination_count = 0;
    for (int i = 0; i < ilu->n_interior; ++i) {
        int global_i = my_lo + i;
        int row_start = U.rowptr[i];
        int row_end   = U.rowptr[i + 1];

        for (int p = row_start; p < row_end; ++p) {
            int c = U.colidx[p];
            if (c >= global_i) break;

            int local_k = c - my_lo;
            assert(local_k >= 0 && local_k < i);

            int diag_k_ptr = diag_ptrs[local_k];
            double U_kk = U.vals[diag_k_ptr];
            assert(U_kk != 0.0 && "Zero pivot in interior factorization");

            U.vals[p] /= U_kk;
            double L_ik = U.vals[p];

            int ptr_i = p + 1;
            int ptr_k = diag_k_ptr + 1;
            int end_k = U.rowptr[local_k + 1];
            int update_count = 0;

            while (ptr_i < row_end && ptr_k < end_k) {
                int col_i = U.colidx[ptr_i];
                int col_k = U.colidx[ptr_k];
                if (col_i == col_k) {
                    U.vals[ptr_i] -= L_ik * U.vals[ptr_k];
                    update_count++;
                    ptr_i++; ptr_k++;
                } else if (col_i < col_k) {
                    ptr_i++;
                } else {
                    ptr_k++;
                }
            }
            elimination_count++;
        }
    }
}

// P must be the permuted matrix: row/column ids need to be consistent across
// ranks for "c < my_lo" to correctly identify cross-rank dependencies.
static void setup_communication_pattern(const CSRMatrix& P, ILUFact* ilu)
{
    int my_lo = ilu->row_offset;

    ilu->needed_global_rows.clear();

    for (int newi = ilu->n_interior; newi < P.nrows; ++newi) {
        for (int k = P.rowptr[newi]; k < P.rowptr[newi + 1]; ++k) {
            int c = P.colidx[k];
            if (c < my_lo) {
                ilu->needed_global_rows.insert(c);
            }
        }
    }

    // Group needed rows by owner rank
    std::map<int, std::vector<int>> need_from;
    for (int g : ilu->needed_global_rows) {
        int owner = owner_of(g, ilu->N, ilu->world_size);
        need_from[owner].push_back(g);
        ilu->recv_from_ranks.insert(owner);
    }

    // Exchange counts
    int ws = ilu->world_size;
    std::vector<int> send_counts(ws, 0), recv_counts(ws, 0);
    for (auto& kv : need_from) send_counts[kv.first] = (int)kv.second.size();
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT, ilu->comm);

    std::vector<int> sdispls(ws, 0), rdispls(ws, 0);
    for (int r = 1; r < ws; ++r) {
        sdispls[r] = sdispls[r-1] + send_counts[r-1];
        rdispls[r] = rdispls[r-1] + recv_counts[r-1];
    }
    int total_send = sdispls[ws-1] + send_counts[ws-1];
    int total_recv = rdispls[ws-1] + recv_counts[ws-1];

    std::vector<int> sbuf(total_send), rbuf(total_recv);
    for (auto& kv : need_from) {
        int r = kv.first;
        std::copy(kv.second.begin(), kv.second.end(), sbuf.begin() + sdispls[r]);
    }
    MPI_Alltoallv(sbuf.data(), send_counts.data(), sdispls.data(), MPI_INT,
                  rbuf.data(), recv_counts.data(), rdispls.data(), MPI_INT,
                  ilu->comm);

    ilu->send_to_ranks.clear();
    for (int r = 0; r < ws; ++r)
        if (recv_counts[r] > 0) ilu->send_to_ranks.insert(r);

    ilu->_send_requests.assign(ws, {});
    for (int r = 0; r < ws; ++r) {
        ilu->_send_requests[r].assign(
            rbuf.begin() + rdispls[r],
            rbuf.begin() + rdispls[r] + recv_counts[r]);
    }
    ilu->_recv_requests = std::move(need_from);
}

// Cache the (static) communication pattern of the upper-triangular part,
// instead of re-negotiating it on every solve/multiply call.
static void setup_backward_pattern(ILUFact* ilu)
{
    const CSRMatrix& LU = ilu->LU;
    int my_lo = ilu->row_offset;
    int n_loc = ilu->local_n;
    int ws    = ilu->world_size;

    std::map<int, std::vector<int>> need;
    for (int newi = 0; newi < n_loc; ++newi) {
        for (int p = LU.rowptr[newi]; p < LU.rowptr[newi + 1]; ++p) {
            int c = LU.colidx[p];
            if (c >= my_lo + n_loc) {
                need[owner_of(c, ilu->N, ilu->world_size)].push_back(c);
            }
        }
    }
    for (auto& kv : need) {
        std::sort(kv.second.begin(), kv.second.end());
        kv.second.erase(std::unique(kv.second.begin(), kv.second.end()), kv.second.end());
    }

    std::vector<int> sn(ws, 0), rn(ws, 0);
    for (auto& kv : need) sn[kv.first] = (int)kv.second.size();
    MPI_Alltoall(sn.data(), 1, MPI_INT, rn.data(), 1, MPI_INT, ilu->comm);

    std::vector<MPI_Request> reqs;
    for (auto& kv : need) {
        MPI_Request rq;
        MPI_Isend(kv.second.data(), (int)kv.second.size(), MPI_INT,
                  kv.first, 420, ilu->comm, &rq);
        reqs.push_back(rq);
    }

    std::map<int, std::vector<int>> serve;
    for (int r = 0; r < ws; ++r) {
        if (rn[r] == 0) continue;
        serve[r].resize(rn[r]);
        MPI_Recv(serve[r].data(), rn[r], MPI_INT, r, 420, ilu->comm, MPI_STATUS_IGNORE);
    }
    if (!reqs.empty())
        MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

    ilu->bwd_need  = std::move(need);
    ilu->bwd_serve = std::move(serve);
}

static void exchange_Uint_rows(ILUFact* ilu)
{
    const CSRMatrix& LU = ilu->LU;
    int my_lo = ilu->row_offset;
    int ws = ilu->world_size;

    struct Bundle {
        std::vector<int>    global_ids;
        std::vector<int>    diag_cols;
        std::vector<int>    nnz_per_row;
        std::vector<int>    cols;
        std::vector<double> vals;
    };
    std::vector<Bundle> sbuf(ws);
    std::vector<int> send_nrows(ws, 0);

    for (int r = 0; r < ws; ++r) {
        for (int g : ilu->_send_requests[r]) {
            // g is already a permuted global row id.
            int newi = g - my_lo;
            if (newi >= ilu->n_interior) continue;

            int s = LU.rowptr[newi], e = LU.rowptr[newi + 1];
            sbuf[r].global_ids.push_back(g);
            sbuf[r].diag_cols.push_back(g);
            sbuf[r].nnz_per_row.push_back(e - s);
            for (int k = s; k < e; ++k) {
                sbuf[r].cols.push_back(LU.colidx[k]);
                sbuf[r].vals.push_back(LU.vals[k]);
            }
            send_nrows[r]++;
        }
    }

    // Exchange row counts
    std::vector<int> recv_nrows(ws, 0);
    MPI_Alltoall(send_nrows.data(), 1, MPI_INT,
                 recv_nrows.data(), 1, MPI_INT, ilu->comm);

    std::vector<MPI_Request> reqs;
    for (int r : ilu->send_to_ranks) {
        if (send_nrows[r] == 0) continue;
        int nr  = send_nrows[r];
        int nnz = (int)sbuf[r].cols.size();
        MPI_Request rq;
        MPI_Isend(sbuf[r].global_ids.data(),  nr,  MPI_INT,    r, 198, ilu->comm, &rq); reqs.push_back(rq);
        MPI_Isend(sbuf[r].diag_cols.data(),   nr,  MPI_INT,    r, 199, ilu->comm, &rq); reqs.push_back(rq);
        MPI_Isend(sbuf[r].nnz_per_row.data(), nr,  MPI_INT,    r, 200, ilu->comm, &rq); reqs.push_back(rq);
        MPI_Isend(sbuf[r].cols.data(),        nnz, MPI_INT,    r, 201, ilu->comm, &rq); reqs.push_back(rq);
        MPI_Isend(sbuf[r].vals.data(),        nnz, MPI_DOUBLE, r, 202, ilu->comm, &rq); reqs.push_back(rq);
    }

    for (auto& kv : ilu->_recv_requests) {
        int src = kv.first;
        int nr  = recv_nrows[src];
        if (nr == 0) continue;

        std::vector<int> gids(nr), diag_cols(nr), nnz_per_row(nr);
        MPI_Recv(gids.data(),        nr, MPI_INT, src, 198, ilu->comm, MPI_STATUS_IGNORE);
        MPI_Recv(diag_cols.data(),   nr, MPI_INT, src, 199, ilu->comm, MPI_STATUS_IGNORE);
        MPI_Recv(nnz_per_row.data(), nr, MPI_INT, src, 200, ilu->comm, MPI_STATUS_IGNORE);

        int total_nnz = 0;
        for (int x : nnz_per_row) total_nnz += x;

        std::vector<int>    cols(total_nnz);
        std::vector<double> vals(total_nnz);
        MPI_Recv(cols.data(), total_nnz, MPI_INT,    src, 201, ilu->comm, MPI_STATUS_IGNORE);
        MPI_Recv(vals.data(), total_nnz, MPI_DOUBLE, src, 202, ilu->comm, MPI_STATUS_IGNORE);

        int off = 0;
        for (int i = 0; i < nr; ++i) {
            int g = gids[i], n = nnz_per_row[i];
            auto& entry = ilu->remote_rows[g];
            entry.diag_col = diag_cols[i];
            entry.cols.assign(cols.begin() + off, cols.begin() + off + n);
            entry.vals.assign(vals.begin() + off, vals.begin() + off + n);
            off += n;
        }
    }

    if (!reqs.empty())
        MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
}

static void exchange_Usep_rows(ILUFact* ilu)
{
    const CSRMatrix& LU = ilu->LU;
    int my_lo = ilu->row_offset;
    int ws    = ilu->world_size;

    // Build send buffers per destination rank.
    struct RowBundle {
        std::vector<int>    global_ids;
        std::vector<int>    diag_cols;
        std::vector<int>    nnz_per_row;
        std::vector<int>    cols;
        std::vector<double> vals;
    };
    std::vector<RowBundle> sbuf(ws);
    std::vector<int> send_nrows(ws, 0);

    for (int r = 0; r < ws; ++r) {
        for (int g : ilu->_send_requests[r]) {
            // g is already a permuted global row id.
            int newi = g - my_lo;
            if (newi < ilu->n_interior) continue;

            int s = LU.rowptr[newi], e = LU.rowptr[newi + 1];
            sbuf[r].global_ids.push_back(g);
            sbuf[r].diag_cols.push_back(g);
            sbuf[r].nnz_per_row.push_back(e - s);
            for (int k = s; k < e; ++k) {
                sbuf[r].cols.push_back(LU.colidx[k]);
                sbuf[r].vals.push_back(LU.vals[k]);
            }
            send_nrows[r]++;
        }
    }

    // Exchange row counts.
    std::vector<int> recv_nrows(ws, 0);
    MPI_Alltoall(send_nrows.data(), 1, MPI_INT,
                 recv_nrows.data(), 1, MPI_INT, ilu->comm);


    // Non-blocking sends
    std::vector<MPI_Request> reqs;
    for (int r : ilu->send_to_ranks) {
        if (send_nrows[r] == 0) continue;
        int nr  = send_nrows[r];
        int nnz = (int)sbuf[r].cols.size();
        MPI_Request rq;
        MPI_Isend(sbuf[r].global_ids.data(),  nr,  MPI_INT,    r, 300, ilu->comm, &rq); reqs.push_back(rq);
        MPI_Isend(sbuf[r].diag_cols.data(),   nr,  MPI_INT,    r, 304, ilu->comm, &rq); reqs.push_back(rq);
        MPI_Isend(sbuf[r].nnz_per_row.data(), nr,  MPI_INT,    r, 301, ilu->comm, &rq); reqs.push_back(rq);
        MPI_Isend(sbuf[r].cols.data(),        nnz, MPI_INT,    r, 302, ilu->comm, &rq); reqs.push_back(rq);
        MPI_Isend(sbuf[r].vals.data(),        nnz, MPI_DOUBLE, r, 303, ilu->comm, &rq); reqs.push_back(rq);
    }

    // Blocking recvs
    for (auto& kv : ilu->_recv_requests) {
        int src = kv.first;
        int nr  = recv_nrows[src];
        if (nr == 0) continue;

        std::vector<int>    gids(nr), diag_cols(nr), nnz_arr(nr);
        MPI_Recv(gids.data(),    nr, MPI_INT, src, 300, ilu->comm, MPI_STATUS_IGNORE);
        MPI_Recv(diag_cols.data(), nr, MPI_INT, src, 304, ilu->comm, MPI_STATUS_IGNORE);
        MPI_Recv(nnz_arr.data(), nr, MPI_INT, src, 301, ilu->comm, MPI_STATUS_IGNORE);

        int total_nnz = 0;
        for (int x : nnz_arr) total_nnz += x;

        std::vector<int>    cols(total_nnz);
        std::vector<double> vals(total_nnz);
        MPI_Recv(cols.data(), total_nnz, MPI_INT,    src, 302, ilu->comm, MPI_STATUS_IGNORE);
        MPI_Recv(vals.data(), total_nnz, MPI_DOUBLE, src, 303, ilu->comm, MPI_STATUS_IGNORE);

        int off = 0;
        for (int i = 0; i < nr; ++i) {
            int g = gids[i], n = nnz_arr[i];
            auto& entry = ilu->remote_rows[g];
            entry.diag_col = diag_cols[i];
            entry.cols.assign(cols.begin() + off, cols.begin() + off + n);
            entry.vals.assign(vals.begin() + off, vals.begin() + off + n);
            off += n;
        }
    }

    if (!reqs.empty())
        MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
}

static void factorize_separator_rows(ILUFact* ilu)
{
    CSRMatrix& LU    = ilu->LU;
    int my_lo       = ilu->row_offset;
    int n_int       = ilu->n_interior;
    int n_local     = LU.nrows;

    // Cache diagonal pointers for interior rows.
    std::vector<int> int_diag(n_int, -1);
    for (int i = 0; i < n_int; ++i) {
        int gi = my_lo + i;
        for (int p = LU.rowptr[i]; p < LU.rowptr[i+1]; ++p)
            if (LU.colidx[p] == gi) { int_diag[i] = p; break; }
    }

    // Retrieve pivot row by column index
    auto get_pivot_row = [&](int c,
                             const int*&    pcols,
                             const double*& pvals,
                             int&           plen,
                             int&           pdiag) -> bool
    {
        if (c >= my_lo && c < my_lo + n_local) {
            int lk   = c - my_lo;
            int pk_s = LU.rowptr[lk];
            int pk_e = LU.rowptr[lk + 1];
            pcols = LU.colidx.data() + pk_s;
            pvals = LU.vals.data()   + pk_s;
            plen  = pk_e - pk_s;
            if (lk < n_int && int_diag[lk] != -1) {
                pdiag = int_diag[lk] - pk_s;
            } else {
                pdiag = -1;
                for (int q = 0; q < plen; ++q)
                    if (pcols[q] == c) { pdiag = q; break; }
            }
            return (pdiag != -1);
        } else {
            auto it = ilu->remote_rows.find(c);
            if (it == ilu->remote_rows.end()) return false;
            pcols = it->second.cols.data();
            pvals = it->second.vals.data();
            plen  = (int)it->second.cols.size();
            int dc = it->second.diag_col;
            pdiag = -1;
            for (int q = 0; q < plen; ++q)
                if (pcols[q] == dc) { pdiag = q; break; }
            return (pdiag != -1);
        }
    };

    for (int newi = n_int; newi < n_local; ++newi) {
        int global_i = my_lo + newi;
        int row_s    = LU.rowptr[newi];
        int row_e    = LU.rowptr[newi + 1];

        for (int p = row_s; p < row_e; ++p) {
            int c = LU.colidx[p];
            if (c >= global_i) break;

            const int*    pcols = nullptr;
            const double* pvals = nullptr;
            int           plen  = 0, pdiag = -1;

            if (!get_pivot_row(c, pcols, pvals, plen, pdiag)) continue;

            double LU_kk = pvals[pdiag];
            if (LU_kk == 0.0) continue;

            LU.vals[p] /= LU_kk;
            double L_ik = LU.vals[p];

            // Sparse row update
            int pi = p + 1;
            int pk = pdiag + 1;
            while (pi < row_e && pk < plen) {
                int ci = LU.colidx[pi], ck = pcols[pk];
                if      (ci == ck) { LU.vals[pi] -= L_ik * pvals[pk]; ++pi; ++pk; }
                else if (ci <  ck) { ++pi; }
                else               { ++pk; }
            }
        }
    }
}

// Convergence check: max change in Usep vals
static double separator_change(const ILUFact* ilu,
                               const std::vector<double>& prev_vals)
{
    const CSRMatrix& LU = ilu->LU;
    double maxd = 0.0;
    int n_int = ilu->n_interior;
    for (int i = n_int; i < LU.nrows; ++i) {
        for (int p = LU.rowptr[i]; p < LU.rowptr[i+1]; ++p) {
            double d = std::fabs(LU.vals[p] - prev_vals[p]);
            if (d > maxd) maxd = d;
        }
    }
    return maxd;
}

struct ILUFact* ILU_factorize(int N, int nnz, const int* row, const int* col, const double* val)
{
    ILUFact* ilu = new ILUFact();
    ilu->comm = MPI_COMM_WORLD;
    MPI_Comm_rank(ilu->comm, &ilu->rank);
    MPI_Comm_size(ilu->comm, &ilu->world_size);

    MPI_Bcast(&N, 1, MPI_INT, 0, ilu->comm);

    CSRMatrix localA;
    distribute_matrix(N, nnz, row, col, val, ilu, localA);

    std::vector<char> is_sep;
    classify_rows(localA, ilu, is_sep);
    build_permutation(is_sep, ilu);
    build_global_permutation(ilu);

    CSRMatrix permA;
    permute_matrix(localA, permA, ilu);

    ilu->LU = permA;

    // Factorize interior
    ILU_factorize_interior(ilu);

    setup_communication_pattern(permA, ilu);
    exchange_Uint_rows(ilu);

    // Iterative separator factorization
    const int    MAX_ITER = 50;
    const double TOL      = 1e-12;

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        std::vector<double> prev_vals(ilu->LU.vals);

        // Exchange current Usep rows with dependent ranks.
        exchange_Usep_rows(ilu);

        // Re-factorize separator rows using Rint + Rsep.
        // First reset separator rows to original A values.
        for (int newi = ilu->n_interior; newi < ilu->LU.nrows; ++newi) {
            int src_s = permA.rowptr[newi], src_e = permA.rowptr[newi + 1];
            int dst_s = ilu->LU.rowptr[newi];
            for (int k = 0; k < src_e - src_s; ++k)
                ilu->LU.vals[dst_s + k] = permA.vals[src_s + k];
        }

        factorize_separator_rows(ilu);

        // Check convergence
        double local_change = separator_change(ilu, prev_vals);
        double global_change = 0.0;
        MPI_Allreduce(&local_change, &global_change, 1, MPI_DOUBLE, MPI_MAX, ilu->comm);

        if (global_change < TOL) {
            break;
        }
    }

    // Build the (static) communication pattern for backward substitution
    // and for the products with U, reused by every ILU_solve/ILU_multiply.
    setup_backward_pattern(ilu);

    return ilu;
}

// ============================================================
//  ILU_solve
// ============================================================
void ILU_solve(struct ILUFact* ilu, const double* b, double* res)
{
    const CSRMatrix& LU = ilu->LU;
    int my_lo = ilu->row_offset;
    int n_loc = ilu->local_n;
    int n_int = ilu->n_interior;

    // Permute b into working vector x
    std::vector<double> x(n_loc);
    for (int newi = 0; newi < n_loc; ++newi)
        x[newi] = b[ilu->new_to_old[newi]];

    // Forward sweep: interior nodes
    for (int newi = 0; newi < n_int; ++newi) {
        int global_i = my_lo + newi;
        for (int p = LU.rowptr[newi]; p < LU.rowptr[newi+1]; ++p) {
            int c = LU.colidx[p];
            if (c >= global_i) break;
            int local_c = c - my_lo;
            double contrib = LU.vals[p] * x[local_c];
            x[newi] -= contrib;
        }
    }

    // Receive forward dependencies from lower ranks. The values arrive in
    // the order of our (sorted) request list, so no ids need to be sent.
    std::unordered_map<int, double> remote_x_fwd;
    for (auto& kv : ilu->_recv_requests) {
        int src = kv.first;
        int nr  = (int)kv.second.size();
        if (nr == 0) continue;

        std::vector<double> xv(nr);
        MPI_Recv(xv.data(), nr, MPI_DOUBLE, src, 401, ilu->comm, MPI_STATUS_IGNORE);
        for (int i = 0; i < nr; ++i) {
            remote_x_fwd[kv.second[i]] = xv[i];
        }
    }

    // Forward sweep: boundary nodes
    for (int newi = n_int; newi < n_loc; ++newi) {
        int global_i = my_lo + newi;
        for (int p = LU.rowptr[newi]; p < LU.rowptr[newi+1]; ++p) {
            int c = LU.colidx[p];
            if (c >= global_i) break;
            double xc;
            if (c >= my_lo && c < my_lo + n_loc) {
                xc = x[c - my_lo];
            } else {
                auto it = remote_x_fwd.find(c);
                xc = (it != remote_x_fwd.end()) ? it->second : 0.0;
            }
            double contrib = LU.vals[p] * xc;
            x[newi] -= contrib;
        }
    }

    // Send completed forward data to higher ranks. Buffers must outlive the
    // MPI_Isend calls below, so they're kept in a vector alongside reqs
    // instead of a loop-local that would go out of scope before Waitall.
    std::vector<MPI_Request> reqs;
    std::vector<std::vector<double>> fwd_bufs;
    for (int r : ilu->send_to_ranks) {
        const std::vector<int>& req = ilu->_send_requests[r];
        if (req.empty()) continue;
        fwd_bufs.emplace_back(req.size());
        for (size_t i = 0; i < req.size(); ++i)
            fwd_bufs.back()[i] = x[req[i] - my_lo];
        MPI_Request rq;
        MPI_Isend(fwd_bufs.back().data(), (int)req.size(), MPI_DOUBLE, r, 401, ilu->comm, &rq);
        reqs.push_back(rq);
    }
    if (!reqs.empty())
        MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

    // Backward sweep: first receive x values for columns owned by higher
    // ranks (they finish their backward sweep before serving us).
    std::unordered_map<int, double> remote_x_bwd;
    for (auto& kv : ilu->bwd_need) {
        int src = kv.first;
        std::vector<double> buf(kv.second.size());
        MPI_Recv(buf.data(), (int)buf.size(), MPI_DOUBLE, src, 411, ilu->comm, MPI_STATUS_IGNORE);
        for (size_t i = 0; i < kv.second.size(); ++i) {
            remote_x_bwd[kv.second[i]] = buf[i];
        }
    }

    // Local backward sweep
    for (int newi = n_loc - 1; newi >= 0; --newi) {
        int global_i = my_lo + newi;
        int diag_p = -1;

        for (int p = LU.rowptr[newi]; p < LU.rowptr[newi+1]; ++p) {
            int c = LU.colidx[p];
            if (c < global_i) continue;
            if (c == global_i) { diag_p = p; continue; }
            double xc;
            if (c < my_lo + n_loc) {
                xc = x[c - my_lo];
            } else {
                auto it = remote_x_bwd.find(c);
                xc = (it != remote_x_bwd.end()) ? it->second : 0.0;
            }
            x[newi] -= LU.vals[p] * xc;
        }
        assert(diag_p != -1 && "Missing diagonal entry");
        x[newi] /= LU.vals[diag_p];
    }

    // Fulfill lower rank requests (send backward dependencies)
    std::vector<MPI_Request> send_reqs;
    std::vector<std::vector<double>> send_bufs;
    for (auto& kv : ilu->bwd_serve) {
        int dst = kv.first;
        send_bufs.emplace_back(kv.second.size());
        for (size_t i = 0; i < kv.second.size(); ++i) {
            send_bufs.back()[i] = x[kv.second[i] - my_lo];
        }
        MPI_Request rq;
        MPI_Isend(send_bufs.back().data(), (int)send_bufs.back().size(),
                  MPI_DOUBLE, dst, 411, ilu->comm, &rq);
        send_reqs.push_back(rq);
    }
    if (!send_reqs.empty()) MPI_Waitall((int)send_reqs.size(), send_reqs.data(), MPI_STATUSES_IGNORE);

    // Un-permute result
    for (int newi = 0; newi < n_loc; ++newi) {
        res[ilu->new_to_old[newi]] = x[newi];
    }
}


// ============================================================
//  ILU_multiply
// ============================================================
void ILU_multiply(struct ILUFact* ilu, const double* b, double* res)
{
    const CSRMatrix& LU = ilu->LU;
    int my_lo = ilu->row_offset;
    int n_loc = ilu->local_n;
    int ws    = ilu->world_size;

    // Permute b into working vector xp
    std::vector<double> xp(n_loc);
    for (int newi = 0; newi < n_loc; ++newi)
        xp[newi] = b[ilu->new_to_old[newi]];

    // Exchange xp entries needed for the upper-triangular product, using the
    // pattern cached at factorization time.
    std::vector<MPI_Request> xp_send_reqs;
    std::vector<std::vector<double>> xp_send_bufs;
    for (auto& kv : ilu->bwd_serve) {
        int dst = kv.first;
        xp_send_bufs.emplace_back(kv.second.size());
        for (size_t i = 0; i < kv.second.size(); ++i) {
            xp_send_bufs.back()[i] = xp[kv.second[i] - my_lo];
        }
        MPI_Request rq;
        MPI_Isend(xp_send_bufs.back().data(), (int)xp_send_bufs.back().size(),
                  MPI_DOUBLE, dst, 501, ilu->comm, &rq);
        xp_send_reqs.push_back(rq);
    }

    std::unordered_map<int, double> remote_xp;
    for (auto& kv : ilu->bwd_need) {
        std::vector<double> buf(kv.second.size());
        MPI_Recv(buf.data(), (int)buf.size(), MPI_DOUBLE, kv.first, 501, ilu->comm, MPI_STATUS_IGNORE);
        for (size_t i = 0; i < kv.second.size(); ++i) {
            remote_xp[kv.second[i]] = buf[i];
        }
    }
    if (!xp_send_reqs.empty()) MPI_Waitall((int)xp_send_reqs.size(), xp_send_reqs.data(), MPI_STATUSES_IGNORE);

    // Multiply by upper triangular part (t = U * xp)
    std::vector<double> t(n_loc, 0.0);
    for (int newi = 0; newi < n_loc; ++newi) {
        int global_i = my_lo + newi;
        for (int p = LU.rowptr[newi]; p < LU.rowptr[newi+1]; ++p) {
            int c = LU.colidx[p];
            if (c < global_i) continue;
            double xc;
            if (c >= my_lo && c < my_lo + n_loc) {
                xc = xp[c - my_lo];
            } else {
                auto it = remote_xp.find(c);
                if (it != remote_xp.end()) {
                    xc = it->second;
                } else {
                    xc = 0.0;
                }
            }
            double contrib = LU.vals[p] * xc;
            t[newi] += contrib;
        }
    }

    // Exchange intermediate t data across ranks
    std::unordered_map<int, double> remote_t;
    std::vector<std::vector<int>>    sg(ws);
    std::vector<std::vector<double>> sv(ws);
    std::vector<int> sn(ws, 0), rn(ws, 0);

    for (int r = 0; r < ws; ++r) {
        for (int g : ilu->_send_requests[r]) {
            sg[r].push_back(g);
            sv[r].push_back(t[g - my_lo]);
            sn[r]++;
        }
    }
    MPI_Alltoall(sn.data(), 1, MPI_INT, rn.data(), 1, MPI_INT, ilu->comm);

    std::vector<MPI_Request> reqs;
    for (int r : ilu->send_to_ranks) {
        if (sn[r] == 0) continue;
        MPI_Request rq;
        MPI_Isend(sg[r].data(), sn[r], MPI_INT,    r, 510, ilu->comm, &rq); reqs.push_back(rq);
        MPI_Isend(sv[r].data(), sn[r], MPI_DOUBLE, r, 511, ilu->comm, &rq); reqs.push_back(rq);
    }
    for (auto& kv : ilu->_recv_requests) {
        int src = kv.first;
        int nr  = rn[src];
        if (nr == 0) continue;
        std::vector<int>    gids(nr);
        std::vector<double> tv(nr);
        MPI_Recv(gids.data(), nr, MPI_INT,    src, 510, ilu->comm, MPI_STATUS_IGNORE);
        MPI_Recv(tv.data(),   nr, MPI_DOUBLE, src, 511, ilu->comm, MPI_STATUS_IGNORE);
        for (int i = 0; i < nr; ++i) {
            remote_t[gids[i]] = tv[i];
        }
    }
    if (!reqs.empty()) MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

    // Multiply by lower triangular part
    std::vector<double> yp(n_loc);
    for (int newi = 0; newi < n_loc; ++newi) {
        int global_i = my_lo + newi;
        double val = t[newi];
        for (int p = LU.rowptr[newi]; p < LU.rowptr[newi+1]; ++p) {
            int c = LU.colidx[p];
            if (c >= global_i) break;
            double tc;
            if (c >= my_lo && c < my_lo + n_loc) {
                tc = t[c - my_lo];
            } else {
                auto it = remote_t.find(c);
                if (it != remote_t.end()) {
                    tc = it->second;
                } else {
                    tc = 0.0;
                }
            }
            double contrib = LU.vals[p] * tc;
            val += contrib;
        }
        yp[newi] = val;
    }

    // Un-permute result
    for (int newi = 0; newi < n_loc; ++newi) {
        res[ilu->new_to_old[newi]] = yp[newi];
    }
}

// ============================================================
//  ILU_free
// ============================================================
void ILU_free(struct ILUFact* ilu)
{
    delete ilu;
}