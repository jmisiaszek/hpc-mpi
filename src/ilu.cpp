#include<algorithm>
#include<iostream>
#include<vector>
#include<mpi.h>
#include<string>
#include<set>
#include<map>

#include "ilu.h"

using namespace std;

struct Entry {
    int row, col;
    double val;

    Entry() : row{0}, col{0}, val{0} {}
    Entry(int _row, int _col, double _val) : row{_row}, col{_col}, val{_val} {}

    bool operator<(const Entry& b) const {
        if (row == b.row) return col < b.col;
        return row < b.row;
    }
};

struct ILUFact {
    int N;
    int first_row, last_row;
    int world_size, rank;
    vector<Entry> entries;
    vector<int> row_start;
    vector<int> row_types;
    vector<Entry> R_int;
    vector<int> perm;
    vector<int> inv_perm;
};

std::ostream& operator<<(std::ostream& os, const Entry& obj) {
    return os << "(" << obj.row << ", " << obj.col << ", " << obj.val << ")";
}

int rank_to_row(int N, int world_size, int rank) {
    return rank * N / world_size;
}

int row_to_rank(int N, int world_size, int row) {
    return ((row + 1) * world_size - 1) / N;
}

// A row is a separator row if it needs data this rank does not own: either it
// has a nonzero in a column owned by a lower rank, or it depends (transitively,
// through the local lower triangle) on a row that does.
//
// NOTE: the transitive closure is what keeps the U-row exchange below correct.
// The sender picks the U part of a requested row using the *permuted* local
// order, while the receiver filters those entries with `e->col <= col` in the
// *original* global numbering; the two only agree while perm is the identity,
// which the closure guarantees (a partition ends up all-interior or
// all-separator for the matrices we run). Dropping the closure -- which would
// be a large win, since it currently pulls the whole local factorization into
// the iterative loop -- first needs that exchange to agree on one numbering.
vector<int> check_int_sep(int start_row, int end_row, vector<Entry>& entries) {
    vector<int> rows(end_row - start_row + 1, 0);

    for (const auto& e : entries) {
        if (e.col < start_row) {
            rows[e.row - start_row] = 1;
        }
    }

    bool changed = true;
    while(changed) {
        changed = false;
        for (const auto& e : entries) {
            if (e.col >= start_row && e.col < e.row) {
                if (rows[e.col - start_row] == 1 && rows[e.row - start_row] == 0) {
                    rows[e.row - start_row] = 1;
                    changed = true;
                }
            }
        }
    }

    return rows;
}

void factorize(
    int target_row, // local idx
    int source_row, // local idx
    int tgt_e_idx,
    vector<Entry>& entries, 
    vector<int>& row_start
) {
    int target_start = row_start[target_row], source_start = row_start[source_row];
    double pivot = 0.0;
    int global_col = -1;
    for (int i = source_start; i < row_start[source_row + 1]; i++) {
        if (entries[i].row == entries[i].col) {
            pivot = entries[i].val;
            global_col = entries[i].col;
            break;
        }
    }

    if (abs(pivot) > 1e-14) {
        entries[tgt_e_idx].val /= pivot;
    }
    double L_ent = entries[tgt_e_idx].val;

    for (int i = source_start; i < row_start[source_row + 1]; i++) {
        int src_col = entries[i].col;
        if (src_col <= global_col) continue;

        for (int j = target_start; j < row_start[target_row + 1]; j++) {
            if (entries[j].col == src_col) {
                entries[j].val -= L_ent * entries[i].val;
                break;
            }
        }
    }
}

struct ILUFact* ILU_factorize(int N, int nnz, const int* row, const int* col, const double* val) {
    int rank;
    int world_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int first_row = rank_to_row(N, world_size, rank);
    int last_row = rank_to_row(N, world_size, rank + 1) - 1;

    vector<Entry> my_entries;

    // 1. Organize data into rank vectors and send to processes.
    if (rank == 0) {
        vector<vector<int>> rank_row(world_size, vector<int>());
        vector<vector<int>> rank_col(world_size, vector<int>());
        vector<vector<double>> rank_val(world_size, vector<double>());

        for (int i = 0; i < nnz; i++) {
            int tgt_rank = row_to_rank(N, world_size, row[i]);
            rank_row[tgt_rank].push_back(row[i]);
            rank_col[tgt_rank].push_back(col[i]);
            rank_val[tgt_rank].push_back(val[i]);
        }

        for (int i = 1; i < world_size; i++) {
            int size = rank_row[i].size();
            MPI_Send(&size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

            MPI_Send(rank_row[i].data(), size, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(rank_col[i].data(), size, MPI_INT, i, 2, MPI_COMM_WORLD);
            MPI_Send(rank_val[i].data(), size, MPI_DOUBLE, i, 3, MPI_COMM_WORLD);
        }

        my_entries.resize(rank_row[0].size());
        for (size_t i = 0; i < rank_row[0].size(); i++) {
            my_entries[i] = Entry(rank_row[0][i], rank_col[0][i], rank_val[0][i]);
        }
    }
    else {
        vector<int> my_row, my_col;
        vector<double> my_val;

        int size;
        MPI_Recv(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        my_row.resize(size);
        my_col.resize(size);
        my_val.resize(size);

        MPI_Recv(my_row.data(), size, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(my_col.data(), size, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(my_val.data(), size, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        my_entries.resize(size);
        for (int i = 0; i < size; i++) {
            my_entries[i] = Entry(my_row[i], my_col[i], my_val[i]);
        }
    }

    // 2. Split into interior / separator groups.
    vector<int> row_types = check_int_sep(first_row, last_row, my_entries);

    // 3. Reorder the rows.
    vector<int> perm(row_types.size());
    vector<int> inv_perm(row_types.size());

    int in_cnt = 0;
    for (size_t i = 0; i < row_types.size(); i++) {
        if (row_types[i] == 0) {
            perm[i] = in_cnt;
            inv_perm[in_cnt] = i;
            in_cnt++;
        }
    }
    for (size_t i = 0; i < row_types.size(); i++) {
        if (row_types[i] == 1) {
            perm[i] = in_cnt;
            inv_perm[in_cnt] = i;
            in_cnt++;
        }
    }

    // Apply the permutation to rows and columns.
    for (size_t i = 0; i < my_entries.size(); i++) {
        my_entries[i].row = perm[my_entries[i].row - first_row] + first_row;
        if (my_entries[i].col >= first_row && my_entries[i].col <= last_row) {
            my_entries[i].col = perm[my_entries[i].col - first_row] + first_row;
        }
    }

    // Permute row types too.
    vector<int> temp(row_types.size(), 0);
    for (size_t i = 0; i < row_types.size(); i++) {
        temp[perm[i]] = row_types[i];
    }
    row_types = temp;

    // 4. Interior rows factorization.
    // Sort the entries according to new rows.
    sort(my_entries.begin(), my_entries.end());
    
    // Indices of first elements for each row.
    vector<int> row_start(last_row - first_row + 2, 0);
    for (auto e : my_entries) {
        row_start[e.row - first_row + 1]++;
    }
    for (size_t i = 1; i < row_start.size(); i++) {
        row_start[i] += row_start[i - 1];
    }

    for (int i = 0; i < (int)row_types.size() && row_types[i] == 0; i++) {
        for (int e_idx = row_start[i]; e_idx < row_start[i + 1]; e_idx++) {
            // Entry e = my_entries[e_idx];
            if (my_entries[e_idx].col < i + first_row) {
                factorize(
                    my_entries[e_idx].row - first_row, 
                    my_entries[e_idx].col - first_row, 
                    e_idx, 
                    my_entries, row_start
                );
            }
        }
    }

    // 5. Send and receive U_int between processes.

    // Find own dependencies.
    map<int, set<int>>dep_rows;
    for (auto& e : my_entries) {
        if (row_types[e.row - first_row] == 1 && e.col < first_row) {
            int owner = row_to_rank(N, world_size, e.col);
            dep_rows[owner].insert(e.col);
        }
    }

    // Send out info on dependencies.
    for (int src = 0; src < rank; src++) {
        vector<int> need(dep_rows[src].begin(), dep_rows[src].end());
        int size = need.size();
        MPI_Send(&size, 1, MPI_INT, src, 0, MPI_COMM_WORLD);
        if (size > 0) {
            MPI_Send(need.data(), size, MPI_INT, src, 1, MPI_COMM_WORLD);
        }
    }

    // Receive deps requests and return data.
    for (int dest = rank + 1; dest < world_size; dest++) {
        int size;
        MPI_Recv(&size, 1, MPI_INT, dest, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        vector<int> requested(size);
        if (size > 0) {
            MPI_Recv(requested.data(), size, MPI_INT, dest, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        vector<Entry> to_send;
        for (int req_row : requested) {
            if (req_row < first_row || req_row > last_row) continue;
            int local_req = req_row - first_row;
            int p_req = perm[local_req];

            if (row_types[p_req] != 0) continue;
            for (int idx = row_start[p_req]; idx < row_start[p_req + 1]; idx++) {
                if (my_entries[idx].col < p_req + first_row) continue;

                Entry e = my_entries[idx];
                e.row = req_row;
                if (e.col >= first_row && e.col <= last_row) {
                    e.col = inv_perm[e.col - first_row] + first_row;
                }
                to_send.push_back(e);
            }
        }

        size = to_send.size();
        MPI_Send(&size, 1, MPI_INT, dest, 2, MPI_COMM_WORLD);
        if (size > 0) {
            MPI_Send(to_send.data(), size * sizeof(Entry), MPI_BYTE, dest, 3, MPI_COMM_WORLD);
        }
    }

    // Receive dependencies data.
    vector<Entry> R_int;
    for (int src = 0; src < rank; src++) {
        int size;
        MPI_Recv(&size, 1, MPI_INT, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (size > 0) {
            vector<Entry> buf(size);
            MPI_Recv(buf.data(), size * sizeof(Entry), MPI_BYTE, src, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (auto& e : buf) {
                if (e.col >= first_row && e.col <= last_row) {
                    e.col = perm[e.col - first_row] + first_row;
                }
            }

            R_int.insert(R_int.end(), buf.begin(), buf.end());
        }
    }

    // Adding my local internal rows to R_int for uniformity.
    for (auto e : my_entries) {
        if (row_types[e.row - first_row] == 0) {
            R_int.push_back(e);
        }
    }

    // 6. Initialize L_sep and U_sep
    vector<Entry> L_sep, U_sep;
    for (auto& e: my_entries) {
        if (row_types[e.row - first_row] == 1) {
            if (e.col < e.row) {
                L_sep.push_back(e);
            }
            else {
                U_sep.push_back(e);
            }
        }
    }

    // 7. Convergence loop
    // Send out row requests.
    for (int src = 0; src < rank; src++) {
        set<int> need;
        for (auto& e : my_entries) {
            if (row_types[e.row - first_row] == 1 && e.col < first_row) {
                if (row_to_rank(N, world_size, e.col) == src) {
                    need.insert(e.col);
                }
            }
        }
        vector<int> tgs(need.begin(), need.end());
        int size = tgs.size();
        MPI_Send(&size, 1, MPI_INT, src, 0, MPI_COMM_WORLD);
        if (size > 0)
            MPI_Send(tgs.data(), size, MPI_INT, src, 1, MPI_COMM_WORLD);
    }

    // Receive incoming requests.
    map<int, vector<int>> incoming_req;
    for (int dest = rank + 1; dest < world_size; dest++) {
        int size;
        MPI_Recv(&size, 1, MPI_INT, dest, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        vector<int> req(size);
        if (size > 0)
            MPI_Recv(req.data(), size, MPI_INT, dest, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        incoming_req[dest] = req;
    }

    // Index of the first separator row: the permutation put every interior row
    // before every separator row, so separator rows (and their entries) are one
    // contiguous tail of my_entries.
    int first_sep = 0;
    while (first_sep < (int)row_types.size() && row_types[first_sep] == 0) first_sep++;
    int sep_start_idx = row_start[first_sep];

    // Original values of the separator entries, index-aligned with my_entries,
    // so each iteration can reset them with a linear pass instead of searching
    // A_sep for every single entry (that search was quadratic in nnz).
    vector<double> A_sep_vals(my_entries.size());
    for (size_t k = 0; k < my_entries.size(); k++) A_sep_vals[k] = my_entries[k].val;
    vector<double> prev_vals(my_entries.size(), 0.0);

    // Pivot rows taken from the interior rows of lower ranks. Those are final
    // by now, so this index is built once instead of once per iteration; only
    // the separator pivot rows change between iterations.
    map<int, vector<Entry*>> pivot_int;
    for (auto& e : R_int) {
        if (e.row < first_row) pivot_int[e.row].push_back(&e);
    }

    while(true) {

        // ---- Post all receives first ----
        vector<int> sizes(rank, 0);
        vector<MPI_Request> size_recv_reqs(rank);
        for (int src = 0; src < rank; src++) {
            MPI_Irecv(&sizes[src], 1, MPI_INT, src, 2, MPI_COMM_WORLD, &size_recv_reqs[src]);
        }

        // ---- Prepare and post all sends ----
        map<int, vector<Entry>> to_send_by_dest;
        vector<MPI_Request> size_send_reqs;
        for (auto& [dest, req] : incoming_req) {
            vector<Entry> to_send;
            for (int req_row : req) {
                if (req_row < first_row || req_row > last_row) continue;
                int local_req = req_row - first_row;
                int p_req = perm[local_req];
                if (row_types[p_req] != 1) continue;
                for (int idx = row_start[p_req]; idx < row_start[p_req + 1]; idx++) {
                    if (my_entries[idx].col < p_req + first_row) continue;
                    Entry e = my_entries[idx];
                    e.row = req_row;
                    if (e.col >= first_row && e.col <= last_row)
                        e.col = inv_perm[e.col - first_row] + first_row;
                    to_send.push_back(e);
                }
            }
            to_send_by_dest[dest] = move(to_send);
        }
        // The size buffers have to stay alive until the matching Waitall below:
        // MPI_Isend only records the pointer, so a per-iteration local would be
        // a dangling read by the time the transfer actually happens.
        vector<int> send_sizes(to_send_by_dest.size());
        {
            int k = 0;
            for (auto& [dest, buf] : to_send_by_dest) {
                send_sizes[k] = (int)buf.size();
                MPI_Request r;
                MPI_Isend(&send_sizes[k], 1, MPI_INT, dest, 2, MPI_COMM_WORLD, &r);
                size_send_reqs.push_back(r);
                k++;
            }
        }

        MPI_Waitall(size_recv_reqs.size(), size_recv_reqs.data(), MPI_STATUSES_IGNORE);

        // ---- Now that we know sizes, post data receives ----
        vector<vector<Entry>> recv_bufs(rank);
        vector<MPI_Request> data_recv_reqs;
        for (int src = 0; src < rank; src++) {
            if (sizes[src] > 0) {
                recv_bufs[src].resize(sizes[src]);
                MPI_Request r;
                MPI_Irecv(recv_bufs[src].data(), sizes[src] * sizeof(Entry), MPI_BYTE,
                        src, 3, MPI_COMM_WORLD, &r);
                data_recv_reqs.push_back(r);
            }
        }

        vector<MPI_Request> data_send_reqs;
        for (auto& [dest, buf] : to_send_by_dest) {
            if (!buf.empty()) {
                MPI_Request r;
                MPI_Isend(buf.data(), buf.size() * sizeof(Entry), MPI_BYTE,
                        dest, 3, MPI_COMM_WORLD, &r);
                data_send_reqs.push_back(r);
            }
        }

        MPI_Waitall(data_recv_reqs.size(), data_recv_reqs.data(), MPI_STATUSES_IGNORE);

        vector<Entry> R_sep;
        for (int src = 0; src < rank; src++) {
            for (auto e : recv_bufs[src]) {
                if (e.col >= first_row && e.col <= last_row)
                    e.col = perm[e.col - first_row] + first_row;
                R_sep.push_back(e);
            }
        }

        // Make sure all our sends have actually completed before we touch/reuse buffers
        MPI_Waitall(size_send_reqs.size(), size_send_reqs.data(), MPI_STATUSES_IGNORE);
        MPI_Waitall(data_send_reqs.size(), data_send_reqs.data(), MPI_STATUSES_IGNORE);

        // Remember the current separator values, then reset them to the
        // original A values before re-factorizing.
        for (size_t k = sep_start_idx; k < my_entries.size(); k++) {
            prev_vals[k] = my_entries[k].val;
            my_entries[k].val = A_sep_vals[k];
        }

        // Factorize sep rows
        map<int, vector<Entry*>> pivot_sep;
        for (auto& e : R_sep) pivot_sep[e.row].push_back(&e);

        for (int i = first_sep; i < (int)row_types.size(); i++) {
            for (int e_idx = row_start[i]; e_idx < row_start[i+1]; e_idx++) {
                int col = my_entries[e_idx].col;
                if (col >= first_row) continue;
                // A remote row is either interior or separator on its owner,
                // so at most one of the two indices holds it.
                const vector<Entry*>* prow_p = nullptr;
                auto it_sep = pivot_sep.find(col);
                if (it_sep != pivot_sep.end()) prow_p = &it_sep->second;
                else {
                    auto it_int = pivot_int.find(col);
                    if (it_int != pivot_int.end()) prow_p = &it_int->second;
                }
                if (prow_p == nullptr) continue;
                const vector<Entry*>& prow = *prow_p;
                double pivot = 0.0;
                for (auto* e : prow) { if (e->col == col) { pivot = e->val; break; } }
                if (abs(pivot) < 1e-14) continue;
                my_entries[e_idx].val /= pivot;
                double L_ent = my_entries[e_idx].val;
                for (auto* e : prow) {
                    if (e->col <= col) continue;
                    for (int iidx = row_start[i]; iidx < row_start[i+1]; iidx++) {
                        if (my_entries[iidx].col == e->col) {
                            my_entries[iidx].val -= L_ent * e->val;
                            break;
                        }
                    }
                }
            }
            for (int e_idx = row_start[i]; e_idx < row_start[i+1]; e_idx++) {
                int col = my_entries[e_idx].col;
                if (col >= first_row && col < i + first_row) {
                    factorize(i, col - first_row, e_idx, my_entries, row_start);
                }
            }
        }

        // Interior entries are untouched by this loop, so only the separator
        // tail can have changed.
        double max_change = 0.0;
        for (size_t k = sep_start_idx; k < my_entries.size(); k++)
            max_change = max(max_change, abs(my_entries[k].val - prev_vals[k]));

        double global_max;
        MPI_Allreduce(&max_change, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (global_max < 1e-9) break;
    }


    struct ILUFact* result = new ILUFact();

    result->N = N;
    result->first_row = first_row;
    result->last_row = last_row;
    result->world_size = world_size;
    result->rank = rank;
    result->entries = my_entries;
    result->row_start = row_start;
    result->row_types = row_types;
    result->R_int = R_int;
    result->perm = perm;
    result->inv_perm = inv_perm;

    return result;
}

void ILU_solve(struct ILUFact* ilu, const double* b, double* res) {
    int N = ilu->N;
    int world_size = ilu->world_size;
    int rank = ilu->rank;
    int first_row = ilu->first_row;
    int last_row = ilu->last_row;
    int local_n = last_row - first_row + 1;
    const vector<Entry>& entries = ilu->entries;
    const vector<int>& row_start = ilu->row_start;
    const vector<int>& row_types = ilu->row_types;
    const vector<int>& perm = ilu->perm;
    const vector<int>& inv_perm = ilu->inv_perm;

    vector<double> y(local_n, 0.0), x(local_n, 0.0);

    // Permute input.
    vector<double> rhs(local_n);
    for (int j = 0; j < local_n; j++) {
        rhs[j] = b[inv_perm[j]];
    }

    // Interior rows, no communication needed.
    int i = 0;
    for (; i < local_n && row_types[i] == 0; i++) {
        double sum = rhs[i];
        for (int idx = row_start[i]; idx < row_start[i + 1]; idx++) {
            int col = entries[idx].col;
            if (col >= first_row && col <= last_row && col < i + first_row) {
                sum -= entries[idx].val * y[col - first_row];
            }
        }
        y[i] = sum;
    }

    // Gather dependencies for separator rows.
    map<int, set<int>> dep_rows;
    for (int r = i; r < local_n; r++) {
        for (int idx = row_start[r]; idx < row_start[r + 1]; idx++) {
            int col = entries[idx].col;
            if (col < first_row) {
                dep_rows[row_to_rank(N, world_size, col)].insert(col);
            }
        }
    }

    // Establish exact message sizes for every rank pair via a single
    // collective, instead of a hand-rolled point-to-point size handshake.
    // This can't stall the way ad-hoc pairwise negotiation can at scale --
    // MPI_Alltoall is implemented internally to make guaranteed progress
    // regardless of rank count or traffic density.
    vector<int> send_counts(world_size, 0);
    vector<vector<int>> send_lists(world_size);
    for (auto& kv : dep_rows) {
        send_lists[kv.first] = vector<int>(kv.second.begin(), kv.second.end());
        send_counts[kv.first] = (int)send_lists[kv.first].size();
    }
    vector<int> recv_counts(world_size, 0);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    vector<MPI_Request> req_send_reqs;
    for (int src = 0; src < rank; src++) {
        if (send_counts[src] > 0) {
            MPI_Request rq;
            MPI_Isend(send_lists[src].data(), send_counts[src], MPI_INT, src, 11, MPI_COMM_WORLD, &rq);
            req_send_reqs.push_back(rq);
        }
    }

    map<int, vector<int>> incoming_requests;
    for (int dest = rank + 1; dest < world_size; dest++) {
        if (recv_counts[dest] > 0) {
            vector<int> req(recv_counts[dest]);
            MPI_Recv(req.data(), recv_counts[dest], MPI_INT, dest, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            incoming_requests[dest] = req;
        }
    }
    if (!req_send_reqs.empty())
        MPI_Waitall((int)req_send_reqs.size(), req_send_reqs.data(), MPI_STATUSES_IGNORE);

    map<int, double> y_ext;
    for (int src = 0; src < rank; src++) {
        if (send_counts[src] > 0) {
            vector<double> vals(send_counts[src]);
            MPI_Recv(vals.data(), send_counts[src], MPI_DOUBLE, src, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int k = 0; k < send_counts[src]; k++) y_ext[send_lists[src][k]] = vals[k];
        }
    }

    for (; i < local_n; i++) {
        double sum = rhs[i];
        for (int idx = row_start[i]; idx < row_start[i + 1]; idx++) {
            int col = entries[idx].col;
            if (col >= first_row && col <= last_row) {
                if (col < i + first_row)
                    sum -= entries[idx].val * y[col - first_row];
            } else if (col < first_row) {
                sum -= entries[idx].val * y_ext[col];
            }
        }
        y[i] = sum;
    }

    vector<MPI_Request> val_send_reqs;
    vector<vector<double>> val_send_bufs;
    for (int dest = rank + 1; dest < world_size; dest++) {
        auto it = incoming_requests.find(dest);
        if (it == incoming_requests.end()) continue;
        val_send_bufs.emplace_back();
        auto& vals = val_send_bufs.back();
        for (int global_row : it->second)
            vals.push_back(y[perm[global_row - first_row]]);
        MPI_Request rq;
        MPI_Isend(vals.data(), (int)vals.size(), MPI_DOUBLE, dest, 12, MPI_COMM_WORLD, &rq);
        val_send_reqs.push_back(rq);
    }
    if (!val_send_reqs.empty())
        MPI_Waitall((int)val_send_reqs.size(), val_send_reqs.data(), MPI_STATUSES_IGNORE);

    map<int, set<int>> dep_rows_b;
    for (int r = 0; r < local_n; r++) {
        for (int idx = row_start[r]; idx < row_start[r + 1]; idx++) {
            int col = entries[idx].col;
            if (col > last_row) {
                dep_rows_b[row_to_rank(N, world_size, col)].insert(col);
            }
        }
    }

    vector<int> send_counts_b(world_size, 0);
    vector<vector<int>> send_lists_b(world_size);
    for (auto& kv : dep_rows_b) {
        send_lists_b[kv.first] = vector<int>(kv.second.begin(), kv.second.end());
        send_counts_b[kv.first] = (int)send_lists_b[kv.first].size();
    }
    vector<int> recv_counts_b(world_size, 0);
    MPI_Alltoall(send_counts_b.data(), 1, MPI_INT, recv_counts_b.data(), 1, MPI_INT, MPI_COMM_WORLD);

    vector<MPI_Request> req_send_reqs_b;
    for (int dest = rank + 1; dest < world_size; dest++) {
        if (send_counts_b[dest] > 0) {
            MPI_Request rq;
            MPI_Isend(send_lists_b[dest].data(), send_counts_b[dest], MPI_INT, dest, 21, MPI_COMM_WORLD, &rq);
            req_send_reqs_b.push_back(rq);
        }
    }

    map<int, vector<int>> incoming_requests_b;
    for (int src = 0; src < rank; src++) {
        if (recv_counts_b[src] > 0) {
            vector<int> req(recv_counts_b[src]);
            MPI_Recv(req.data(), recv_counts_b[src], MPI_INT, src, 21, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            incoming_requests_b[src] = req;
        }
    }
    if (!req_send_reqs_b.empty())
        MPI_Waitall((int)req_send_reqs_b.size(), req_send_reqs_b.data(), MPI_STATUSES_IGNORE);

    map<int, double> x_ext;
    for (int dest = rank + 1; dest < world_size; dest++) {
        if (send_counts_b[dest] > 0) {
            vector<double> vals(send_counts_b[dest]);
            MPI_Recv(vals.data(), send_counts_b[dest], MPI_DOUBLE, dest, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int k = 0; k < send_counts_b[dest]; k++) x_ext[send_lists_b[dest][k]] = vals[k];
        }
    }

    for (int r = local_n - 1; r >= i; r--) {
        double sum = y[r];
        double diag = 1.0;
        for (int idx = row_start[r]; idx < row_start[r + 1]; idx++) {
            int col = entries[idx].col;
            if (col == r + first_row) {
                diag = entries[idx].val;
            } else if (col > r + first_row && col <= last_row) {
                sum -= entries[idx].val * x[col - first_row];
            } else if (col > last_row) {
                sum -= entries[idx].val * x_ext[col];
            }
        }
        x[r] = sum / diag;
    }

    for (int r = i - 1; r >= 0; r--) {
        double sum = y[r];
        double diag = 1.0;
        for (int idx = row_start[r]; idx < row_start[r + 1]; idx++) {
            int col = entries[idx].col;
            if (col == r + first_row) {
                diag = entries[idx].val;
            } else if (col > r + first_row && col <= last_row) {
                sum -= entries[idx].val * x[col - first_row];
            } else if (col > last_row) {
                sum -= entries[idx].val * x_ext[col];
            }
        }
        x[r] = sum / diag;
    }

    vector<MPI_Request> val_send_reqs_b;
    vector<vector<double>> val_send_bufs_b;
    for (int src = 0; src < rank; src++) {
        auto it = incoming_requests_b.find(src);
        if (it == incoming_requests_b.end()) continue;
        val_send_bufs_b.emplace_back();
        auto& vals = val_send_bufs_b.back();
        for (int global_row : it->second)
            vals.push_back(x[perm[global_row - first_row]]);
        MPI_Request rq;
        MPI_Isend(vals.data(), (int)vals.size(), MPI_DOUBLE, src, 22, MPI_COMM_WORLD, &rq);
        val_send_reqs_b.push_back(rq);
    }
    if (!val_send_reqs_b.empty())
        MPI_Waitall((int)val_send_reqs_b.size(), val_send_reqs_b.data(), MPI_STATUSES_IGNORE);

    for (int local_old = 0; local_old < local_n; local_old++) {
        res[local_old] = x[perm[local_old]];
    }
}

void ILU_multiply(struct ILUFact* ilu, const double* b, double* res) {
    int N = ilu->N;
    int world_size = ilu->world_size;
    int rank = ilu->rank;
    int first_row = ilu->first_row;
    int last_row = ilu->last_row;
    int local_n = last_row - first_row + 1;

    const vector<Entry>& entries = ilu->entries;
    const vector<int>& row_start = ilu->row_start;
    const vector<int>& perm = ilu->perm;

    // 1. U * b
    map<int, set<int>> u_dep_rows, l_dep_rows;
    for (int r = 0; r < local_n; r++) {
        for (int idx = row_start[r]; idx < row_start[r + 1]; idx++) {
            int col = entries[idx].col, row = entries[idx].row;
            if (col >= row) { // upper
                if (col > last_row) {
                    u_dep_rows[row_to_rank(N, world_size, col)].insert(col);
                }
            }
            if (col < row) { // lower
                if (col < first_row) {
                    l_dep_rows[row_to_rank(N, world_size, col)].insert(col);
                }
            }
        }
    }

    map<int, double> ext_vals;
    {
        // Establish exact per-rank message sizes via one collective, up
        // front, instead of a hand-rolled point-to-point size handshake
        // (tag 10) done independently by every rank.
        vector<vector<int>> send_bufs(world_size);
        vector<int> send_counts(world_size, 0);
        for (int src = 0; src < world_size; src++) {
            if (src == rank) continue;
            if (src > rank) {
                send_bufs[src] = vector<int>(u_dep_rows[src].begin(), u_dep_rows[src].end());
            } else {
                send_bufs[src] = vector<int>(l_dep_rows[src].begin(), l_dep_rows[src].end());
            }
            send_counts[src] = (int)send_bufs[src].size();
        }

        vector<int> recv_counts(world_size, 0);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        vector<MPI_Request> ask_reqs, ans_reqs;
        for (int src = 0; src < world_size; src++) {
            if (src == rank) continue;
            if (send_counts[src] > 0) {
                MPI_Request r2;
                MPI_Isend(send_bufs[src].data(), send_counts[src], MPI_INT, src, 11, MPI_COMM_WORLD, &r2);
                ask_reqs.push_back(r2);
            }
        }

        vector<vector<double>> ans_bufs(world_size);
        for (int dest = 0; dest < world_size; dest++) {
            if (dest == rank) continue;
            if (recv_counts[dest] > 0) {
                vector<int> req(recv_counts[dest]);
                MPI_Recv(req.data(), recv_counts[dest], MPI_INT, dest, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (auto col : req) {
                    int local_col = col - first_row;
                    ans_bufs[dest].push_back(b[local_col]);
                }
                MPI_Request r2;
                MPI_Isend(ans_bufs[dest].data(), recv_counts[dest], MPI_DOUBLE, dest, 1, MPI_COMM_WORLD, &r2);
                ans_reqs.push_back(r2);
            }
        }

        if (!ask_reqs.empty())
            MPI_Waitall((int)ask_reqs.size(), ask_reqs.data(), MPI_STATUSES_IGNORE);

        for (int src = 0; src < world_size; src++) {
            if (src == rank) continue;
            if (send_counts[src] > 0) {
                vector<double> vals(send_counts[src]);
                MPI_Recv(vals.data(), send_counts[src], MPI_DOUBLE, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int i = 0; i < send_counts[src]; i++) {
                    ext_vals[send_bufs[src][i]] = vals[i];
                }
            }
        }

        if (!ans_reqs.empty())
            MPI_Waitall((int)ans_reqs.size(), ans_reqs.data(), MPI_STATUSES_IGNORE);
    }

    for (int i = 0; i < local_n; i++) {
        ext_vals[perm[i] + first_row] = b[i];
    }

    vector<double> partial(local_n, 0);
    for (int i = 0; i < local_n; i++) {
        int global_row = i + first_row;
        for (int idx = row_start[i]; idx < row_start[i + 1]; idx++) {
            if (entries[idx].col >= global_row) {
                partial[i] += entries[idx].val * ext_vals[entries[idx].col];
            }
        }
    }

    {
        vector<vector<int>> send_bufs(world_size);
        vector<int> send_counts(world_size, 0);
        for (int src = 0; src < world_size; src++) {
            if (src == rank) continue;
            if (src > rank) {
                send_bufs[src] = vector<int>(u_dep_rows[src].begin(), u_dep_rows[src].end());
            } else {
                send_bufs[src] = vector<int>(l_dep_rows[src].begin(), l_dep_rows[src].end());
            }
            send_counts[src] = (int)send_bufs[src].size();
        }

        vector<int> recv_counts(world_size, 0);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        vector<MPI_Request> ask_reqs, ans_reqs;
        for (int src = 0; src < world_size; src++) {
            if (src == rank) continue;
            if (send_counts[src] > 0) {
                MPI_Request r2;
                MPI_Isend(send_bufs[src].data(), send_counts[src], MPI_INT, src, 11, MPI_COMM_WORLD, &r2);
                ask_reqs.push_back(r2);
            }
        }

        vector<vector<double>> ans_bufs(world_size);
        for (int dest = 0; dest < world_size; dest++) {
            if (dest == rank) continue;
            if (recv_counts[dest] > 0) {
                vector<int> req(recv_counts[dest]);
                MPI_Recv(req.data(), recv_counts[dest], MPI_INT, dest, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (auto col : req) {
                    int local_col = col - first_row;
                    ans_bufs[dest].push_back(partial[perm[local_col]]);
                }
                MPI_Request r2;
                MPI_Isend(ans_bufs[dest].data(), recv_counts[dest], MPI_DOUBLE, dest, 1, MPI_COMM_WORLD, &r2);
                ans_reqs.push_back(r2);
            }
        }

        if (!ask_reqs.empty())
            MPI_Waitall((int)ask_reqs.size(), ask_reqs.data(), MPI_STATUSES_IGNORE);

        // recv data
        ext_vals.clear();
        for (int src = 0; src < world_size; src++) {
            if (src == rank) continue;
            if (send_counts[src] > 0) {
                vector<double> vals(send_counts[src]);
                MPI_Recv(vals.data(), send_counts[src], MPI_DOUBLE, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int i = 0; i < send_counts[src]; i++) {
                    ext_vals[send_bufs[src][i]] = vals[i];
                }
            }
        }

        if (!ans_reqs.empty())
            MPI_Waitall((int)ans_reqs.size(), ans_reqs.data(), MPI_STATUSES_IGNORE);
    }

    for (int i = 0; i < local_n; i++) {
        ext_vals[i + first_row] = partial[i];
    }

    vector<double> final(local_n, 0);
    for (int i = 0; i < local_n; i++) {
        int global_row = i + first_row;
        for (int idx = row_start[i]; idx < row_start[i + 1]; idx++) {
            if (entries[idx].col < global_row) {
                final[i] += entries[idx].val * ext_vals[entries[idx].col];
            }
        }
        final[i] += ext_vals[global_row];
    }

    for (int local_old = 0; local_old < local_n; local_old++) {
        res[local_old] = final[perm[local_old]];
    }
}

void ILU_free(struct ILUFact* ilu) {
    delete ilu;
}