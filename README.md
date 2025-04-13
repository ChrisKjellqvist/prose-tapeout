# ProSE Tapeout

This work is the implementation of _ProSE: the architecture and design of a protein discovery engine_ [[ASPLOS '22]](https://dl.acm.org/doi/pdf/10.1145/3503222.3507722?casa_token=AaaaUhe5T24AAAAA:mbtrAIdZRIy1PcjeEgdtGXarFZ29ODWcVOj2iyeQ2JA1WP7y4dMR40AFGfMu6DJg0e1g1pcGNAcFVQ). 
When we decided to tape out this accelerator, Protein discovery via BERT models was no longer the primary goal. Instead, with the recent introduction
of LLMs, which shared the dependence on long context lengths with [ProteinBERT](https://github.com/nadavbra/protein_bert),
we decided to retarget the ProSE architecture towards LLMs. ProSE is all about maintaining high utilization
of systolic arrays under memory bandwidth limitations so we thought it would be a good fit.

### High-Level Idea

[Roofline models](https://en.wikipedia.org/wiki/Roofline_model) are a way of modeling bottlenecks in throughput-oriented
systems and showing this "regime" change between memory bound and computationally bound programs. Theoretically then,
given the Bytes-per-OP for a program, we can know roughly how it will behave on the architecture. For the case of
output stationary systolic arrays, we can look at the following plot.

![](img/roofline-alone.jpg)
