#pragma once

#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace grav_mpi {

inline void check_mpi(int rc, const char* call) {
  if (rc != MPI_SUCCESS) {
    char err[MPI_MAX_ERROR_STRING]{};
    int len = 0;
    MPI_Error_string(rc, err, &len);
    throw std::runtime_error(std::string(call) + " failed: " + std::string(err, len));
  }
}

#define GRAV_MPI_CHECK(x) ::grav_mpi::check_mpi((x), #x)

struct GravMpiCtx {
  MPI_Comm world = MPI_COMM_NULL;
  MPI_Comm node_comm = MPI_COMM_NULL;
  int world_rank = 0;
  int world_size = 1;
  int local_rank = 0;
  int local_size = 1;
  int node_id = 0;
  int num_nodes = 1;
  int cuda_dev = 0;
  bool cuda_aware = false;
  std::vector<int> rank_to_node;
  std::string proc_name;

  void init(MPI_Comm comm = MPI_COMM_WORLD) {
    world = comm;
    GRAV_MPI_CHECK(MPI_Comm_rank(world, &world_rank));
    GRAV_MPI_CHECK(MPI_Comm_size(world, &world_size));
    GRAV_MPI_CHECK(MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm));
    GRAV_MPI_CHECK(MPI_Comm_rank(node_comm, &local_rank));
    GRAV_MPI_CHECK(MPI_Comm_size(node_comm, &local_size));

    char name[MPI_MAX_PROCESSOR_NAME]{};
    int nlen = 0;
    GRAV_MPI_CHECK(MPI_Get_processor_name(name, &nlen));
    proc_name.assign(name, nlen);

    // Gather truncated processor names to assign deterministic 0..num_nodes-1 node_ids
    constexpr int kNameStride = 64;
    std::vector<char> my_name(kNameStride, '\0');
    std::memcpy(my_name.data(), proc_name.data(), std::min<size_t>(proc_name.size(), kNameStride - 1));
    std::vector<char> all_names(size_t(world_size) * kNameStride, '\0');
    GRAV_MPI_CHECK(MPI_Allgather(my_name.data(), kNameStride, MPI_CHAR,
                                 all_names.data(), kNameStride, MPI_CHAR, world));

    std::map<std::string, int> name_to_id;
    rank_to_node.resize(world_size, 0);
    for (int r = 0; r < world_size; ++r) {
      std::string rn(&all_names[size_t(r) * kNameStride]);
      auto it = name_to_id.find(rn);
      if (it == name_to_id.end()) {
        int id = int(name_to_id.size());
        name_to_id[rn] = id;
        rank_to_node[r] = id;
      } else {
        rank_to_node[r] = it->second;
      }
    }
    node_id = rank_to_node[world_rank];
    num_nodes = int(name_to_id.size());

    int dev_count = 0;
    cudaError_t ce = cudaGetDeviceCount(&dev_count);
    if (ce != cudaSuccess || dev_count <= 0) {
      throw std::runtime_error("No CUDA device visible to MPI rank " + std::to_string(world_rank));
    }
    cuda_dev = (dev_count == 1) ? 0 : (local_rank % dev_count);
    ce = cudaSetDevice(cuda_dev);
    if (ce != cudaSuccess) {
      throw std::runtime_error("cudaSetDevice failed on rank " + std::to_string(world_rank) +
                               ": " + cudaGetErrorString(ce));
    }
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    const char* ca_env = std::getenv("GRAV_MPI_CUDA_AWARE");
    cuda_aware = (ca_env && std::string(ca_env) == "1");
  }

  void destroy() {
    if (node_comm != MPI_COMM_NULL) {
      MPI_Comm_free(&node_comm);
      node_comm = MPI_COMM_NULL;
    }
  }

  bool same_node(int r1, int r2) const {
    return rank_to_node[r1] == rank_to_node[r2];
  }

  // Exchange variable-size byte chunks across all ranks using counts-per-peer.
  // send_counts[dst] is in bytes; recv_counts[src] is filled in bytes.
  // If cuda_aware==false, caller passes host pointers; helper supports device pointers directly too.
  void alltoallv_bytes_dev(
      const void* d_send, const std::vector<int>& send_bytes,
      void* d_recv, std::vector<int>& recv_bytes,
      std::vector<int>& recv_displs,
      cudaStream_t stream,
      std::vector<char>& h_send_buf,
      std::vector<char>& h_recv_buf) const {
    recv_bytes.assign(world_size, 0);
    GRAV_MPI_CHECK(MPI_Alltoall(send_bytes.data(), 1, MPI_INT,
                                recv_bytes.data(), 1, MPI_INT, world));

    std::vector<int> sdispls(world_size, 0);
    recv_displs.assign(world_size, 0);
    int tot_send = 0, tot_recv = 0;
    for (int r = 0; r < world_size; ++r) {
      sdispls[r] = tot_send;
      recv_displs[r] = tot_recv;
      tot_send += send_bytes[r];
      tot_recv += recv_bytes[r];
    }

    if (cuda_aware) {
      if (tot_send > 0 || tot_recv > 0) {
        cudaStreamSynchronize(stream);
        GRAV_MPI_CHECK(MPI_Alltoallv(
            d_send, send_bytes.data(), sdispls.data(), MPI_BYTE,
            d_recv, recv_bytes.data(), recv_displs.data(), MPI_BYTE, world));
      }
      return;
    }

    h_send_buf.resize(size_t(std::max(1, tot_send)));
    h_recv_buf.resize(size_t(std::max(1, tot_recv)));
    if (tot_send > 0 && d_send) {
      cudaError_t ce = cudaMemcpyAsync(h_send_buf.data(), d_send, size_t(tot_send),
                                       cudaMemcpyDeviceToHost, stream);
      if (ce != cudaSuccess) throw std::runtime_error(cudaGetErrorString(ce));
    }
    cudaStreamSynchronize(stream);
    if (tot_send > 0 || tot_recv > 0) {
      GRAV_MPI_CHECK(MPI_Alltoallv(
          h_send_buf.data(), send_bytes.data(), sdispls.data(), MPI_BYTE,
          h_recv_buf.data(), recv_bytes.data(), recv_displs.data(), MPI_BYTE, world));
    }
    if (tot_recv > 0 && d_recv) {
      cudaError_t ce = cudaMemcpyAsync(d_recv, h_recv_buf.data(), size_t(tot_recv),
                                       cudaMemcpyHostToDevice, stream);
      if (ce != cudaSuccess) throw std::runtime_error(cudaGetErrorString(ce));
      cudaStreamSynchronize(stream);
    }
  }

  // Sparse non-blocking point-to-point exchange of device buffers.
  // Sends send_ptrs[i] (send_bytes[i] > 0) to send_peers[i] and receives into
  // recv_ptrs[j] (recv_bytes[j] > 0) from recv_peers[j].
  void exchange_sparse_dev(
      const std::vector<int>& send_peers,
      const std::vector<const void*>& send_ptrs,
      const std::vector<int>& send_bytes,
      std::vector<std::vector<char>>& h_send_scratch,
      const std::vector<int>& recv_peers,
      const std::vector<void*>& recv_ptrs,
      const std::vector<int>& recv_bytes,
      std::vector<std::vector<char>>& h_recv_scratch,
      int tag,
      cudaStream_t stream) const {
    const size_t ns = send_peers.size();
    const size_t nr = recv_peers.size();
    if (ns == 0 && nr == 0) return;

    std::vector<MPI_Request> reqs;
    reqs.reserve(ns + nr);

    if (cuda_aware) {
      cudaStreamSynchronize(stream);
      for (size_t j = 0; j < nr; ++j) {
        if (recv_bytes[j] <= 0) continue;
        MPI_Request req = MPI_REQUEST_NULL;
        GRAV_MPI_CHECK(MPI_Irecv(recv_ptrs[j], recv_bytes[j], MPI_BYTE,
                                 recv_peers[j], tag, world, &req));
        reqs.push_back(req);
      }
      for (size_t i = 0; i < ns; ++i) {
        if (send_bytes[i] <= 0) continue;
        MPI_Request req = MPI_REQUEST_NULL;
        GRAV_MPI_CHECK(MPI_Isend(send_ptrs[i], send_bytes[i], MPI_BYTE,
                                 send_peers[i], tag, world, &req));
        reqs.push_back(req);
      }
      if (!reqs.empty()) {
        GRAV_MPI_CHECK(MPI_Waitall(int(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE));
      }
      return;
    }

    h_send_scratch.resize(ns);
    h_recv_scratch.resize(nr);
    for (size_t i = 0; i < ns; ++i) {
      if (send_bytes[i] <= 0) continue;
      h_send_scratch[i].resize(size_t(send_bytes[i]));
      cudaError_t ce = cudaMemcpyAsync(h_send_scratch[i].data(), send_ptrs[i],
                                       size_t(send_bytes[i]), cudaMemcpyDeviceToHost, stream);
      if (ce != cudaSuccess) throw std::runtime_error(cudaGetErrorString(ce));
    }
    for (size_t j = 0; j < nr; ++j) {
      if (recv_bytes[j] <= 0) continue;
      h_recv_scratch[j].resize(size_t(recv_bytes[j]));
      MPI_Request req = MPI_REQUEST_NULL;
      GRAV_MPI_CHECK(MPI_Irecv(h_recv_scratch[j].data(), recv_bytes[j], MPI_BYTE,
                               recv_peers[j], tag, world, &req));
      reqs.push_back(req);
    }
    if (ns > 0) cudaStreamSynchronize(stream);
    for (size_t i = 0; i < ns; ++i) {
      if (send_bytes[i] <= 0) continue;
      MPI_Request req = MPI_REQUEST_NULL;
      GRAV_MPI_CHECK(MPI_Isend(h_send_scratch[i].data(), send_bytes[i], MPI_BYTE,
                               send_peers[i], tag, world, &req));
      reqs.push_back(req);
    }
    if (!reqs.empty()) {
      GRAV_MPI_CHECK(MPI_Waitall(int(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE));
    }
    for (size_t j = 0; j < nr; ++j) {
      if (recv_bytes[j] <= 0) continue;
      cudaError_t ce = cudaMemcpyAsync(recv_ptrs[j], h_recv_scratch[j].data(),
                                       size_t(recv_bytes[j]), cudaMemcpyHostToDevice, stream);
      if (ce != cudaSuccess) throw std::runtime_error(cudaGetErrorString(ce));
    }
    if (nr > 0) cudaStreamSynchronize(stream);
  }
};

} // namespace grav_mpi
