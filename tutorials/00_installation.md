To install ParaGen, 
- build a basic running environment via `./Dockerfile`
- run `pip3 install -e .`

# ParaGen Commands

After build wheels of ParaGen, follows commands can be used to run ParaGen in different mode.
- `paragen-run`: train and evaluate ParaGen tasks.         
- `paragen-build-tokenizer`: build tokenizer vocabulary and related resources to encode data.
- `paragen-preprocess`: preprocess data with task-specific data processing function without creating batches.
- `paragen-binarize-data`: pre-creating batches for data
- `paragen-export`: export neural model in `torch.jit` or `lightseq`
- `paragen-serve`: run ParaGen task as a service with only GPU-free processing
- `paragen-serve-model`: run ParaGen task as a serivce with only GPU processing
