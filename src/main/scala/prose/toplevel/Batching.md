Batching is hard.
We want to hide batching inside the FPU latency by pipelining the FPU FMA. IF the specified FPU latency
is less than the maximum supported batching, then it's not entirely clear that there's a downside to
increasing the latency of the FPU. On the contrary, it would make it easier to synthesize/place/route.
However, the FPU latency does provide a lower boundary on a reasonable minimum for batching.

For instance, consider batchSize=1 and FPU latency = k>1. Then during compute we're actually wasting
compute on the order of 1/(k).

Number of streams:
An additional consideration is how we're doing the memory layout. The current idea is to do the memory
layout such that it aligns nicely with the minimum size array in the design. Consider two matrices of
dimension J and K for K>J and they're both powers of 2. Whenever systolic array K is reading in from a
single address it's only going to get J/K of the memory it needs, requiring K/J streams to actually 
logically read in a contiguous row.

