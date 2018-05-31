__kernel void blellochScan(
	__global const int * input,
	__global int *output,
	const int n)
{
	__global int *temp = output;
	int thid = get_local_id(0);
	int offset = 1;

	temp[2 * thid] = input[2 * thid];
	temp[2 * thid + 1] = input[2 * thid + 1];

	for (int d = n >> 1; d > 0; d >>= 1) {
		barrier(CLK_LOCAL_MEM_FENCE);
		if (thid < d) {
			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset * (2 * thid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (thid == 0) { temp[n - 1] = 0; }

	for (int d = 1; d < n; d *= 2) {
		offset >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE);
		if (thid < d) {
			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset * (2 * thid + 2) - 1;
			//swap and add
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	output[2 * thid] = temp[2 * thid];
	output[2 * thid + 1] = temp[2 * thid + 1];
}