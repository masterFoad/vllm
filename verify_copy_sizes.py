import numpy as np


def simulate_old_copy(num_reqs, num_tokens_per_req, num_reqs_padded, max_num_reqs=1024):
    query_start_loc_np = np.empty(max_num_reqs + 1, dtype=np.int32)
    query_start_loc_np[0] = 0
    np.cumsum(num_tokens_per_req, out=query_start_loc_np[1 : num_reqs + 1])
    query_start_loc_np[num_reqs + 1 :] = sum(num_tokens_per_req)

    # Old logic copies the ENTIRE max_num_reqs + 1 buffer
    copy_size = query_start_loc_np.shape[0]
    return copy_size


def simulate_new_copy(num_reqs, num_tokens_per_req, num_reqs_padded, max_num_reqs=1024):
    # New logic explicitly limits the buffer slice
    query_start_loc_np = np.empty(max_num_reqs + 1, dtype=np.int32)
    query_start_loc_np = query_start_loc_np[: num_reqs_padded + 1]

    query_start_loc_np[0] = 0
    np.cumsum(num_tokens_per_req, out=query_start_loc_np[1 : num_reqs + 1])
    if num_reqs_padded > num_reqs:
        query_start_loc_np[num_reqs + 1 :] = sum(num_tokens_per_req)

    # New logic copies ONLY the active/padded slice
    copy_size = query_start_loc_np.shape[0]
    return copy_size


if __name__ == "__main__":
    num_reqs = 5
    num_reqs_padded = 16
    tokens = np.ones(num_reqs, dtype=np.int32)

    print("For a batch size of 5 requests (padded to 16):")
    print(
        f"Old copy size (elements): {simulate_old_copy(num_reqs, tokens, num_reqs_padded)}"
    )
    print(
        f"New copy size (elements): {simulate_new_copy(num_reqs, tokens, num_reqs_padded)}"
    )
