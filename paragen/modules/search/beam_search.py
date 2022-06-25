import torch
import torch.nn.functional as F

from paragen.modules.search import register_search
from paragen.modules.search.sequence_search import SequenceSearch
from paragen.modules.utils import create_init_scores
from paragen.utils.ops import recursive


@register_search
class BeamSearch(SequenceSearch):
    """
    BeamSearch is beam search on sequence generation.

    Args:
        beam: beam size
        lenpen: length penalty
        maxlen_coef (a, b): maxlen computation coefficient.
            The max length is computed as `(S * a + b)`, where S is source sequence length.
        maxlen: maximum length
        num_return_sequence: the number of return generated sequence for each input sample,
            which must smaller or equal to beam size, default as 1 to return the sequence
            with the highest log probability
        keepdim: bool, squeeze the output sequence if set True
    """

    def __init__(self, beam=4, lenpen=0.1, maxlen_coef=(1.2, 10), minlen=100, maxlen=1000, num_return_sequence=1, keepdim=False):
        super().__init__()
        self._beam = beam
        self._lenpen = lenpen
        self._minlen = minlen
        self._maxlen = maxlen
        self._maxlen_a, self._maxlen_b = maxlen_coef
        self._num_return_sequence = num_return_sequence
        self._keepdim = keepdim
        self._expand_cache = recursive(self._expand_tensor)
        self._update_cache = recursive(self._update_tensor)

        assert self._num_return_sequence <= self._beam, \
            f"number of return sequence <= beam size, got {self.num_return_sequence} > {self._beam}"

    def forward(self,
                tokens,
                memory,
                memory_padding_mask,
                target_mask=None,
                prev_scores=None):
        """
        Decoding full-step sequence with beam search

        Args:
            tokens: previous tokens or prefix of sequence
              :math:`(N, T)` where N is batch size and T is current sequence length
            memory: memory for attention.
              :math:`(M, N, E)`, where M is the memory sequence length, N is the batch size,
            memory_padding_mask: memory sequence padding mask.
              :math:`(N, M)` where M is the memory sequence length, N is the batch size.
            target_mask: target mask indicating blacklist tokens
              :math:`(N, V)` where N is batch size and V is vocab size
            prev_scores: scores of previous tokens
              :math:`(N)` where N is batch size

        Returns:
            - log probability of generated sequence
                :math: `(N, S)` where N is batch size and S is the number of return sequence
            - generated sequence
                :math: `(N, S, maxT)` where N is batch size, maxT is the maximum sequence length
                    and S is the number of return sequence
                       `(N, maxT)` if self.keepdim=True and S = 1
        """
        batch_size, curlen = tokens.size(0), tokens.size(1)
        memory_size = memory_padding_mask.size(-1)
        scores = create_init_scores(tokens, memory) if prev_scores is None else prev_scores
        # produce the first token
        logits = self._decoder(tokens, memory,
                               memory_padding_mask=memory_padding_mask)[:, -1, :]
        logits = F.log_softmax(logits, dim=-1)
        next_token_score, next_token = logits.topk(self._beam, dim=-1)
        scores = scores.unsqueeze(-1) + next_token_score
        next_token = next_token.unsqueeze(dim=-1)
        tokens = torch.cat([tokens.unsqueeze(dim=1).repeat(1, self._beam, 1), next_token], dim=-1)

        # copy cache and memory for 'beam' times
        memory, memory_padding_mask = self._expand(memory, memory_padding_mask)

        # generate rest of tokens
        maxlen = min(int(memory_size * self._maxlen_a + self._maxlen_b) - curlen, self._maxlen)
        finished = tokens.data.new(batch_size, 1, maxlen + curlen + 1).fill_(self._eos)
        finished_scores = scores.data.new(batch_size, 1).fill_(float('-inf'))
        for l in range(maxlen - 1):
            tokens, scores = self._produce_candidates(tokens, memory, memory_padding_mask, scores)
            finished_mask = tokens[:, :, -1] == self._eos
            if l > self._minlen:
                finished, finished_scores = self._add_finished(finished, finished_scores, tokens, scores, finished_mask)
            tokens, scores = self._update_states(tokens, scores, finished_mask)

        if not self._keepdim and self._num_return_sequence == 1:
            finished = finished.squeeze(dim=1)
        return finished_scores, finished

    def _produce_candidates(self, tokens, memory, memory_padding_mask, scores):
        """
        Predict `beam**2 * 2` candidates at next step

        Args:
            tokens: previous tokens or prefix of sequence at time step T
              :math:`(N, B, T)` where N is batch size, B is beam size and T is current sequence length
            memory: memory for attention.
              :math:`(M, N * B, E)`, where M is the memory sequence length, N is the batch size,
              B is beam size and E is feature size
            memory_padding_mask: memory sequence padding mask.
              :math:`(N * B, M)` where M is the memory sequence length, N is the batch size,
              B is beam size.
            scores: scores of previous tokens
              :math:`(N * B)` where N is batch size and B is beam size.

        Returns:
            - candidate tokens at time step T+1
              :math:`(N, B * B * 2, T+1)` where N is batch size, B is beam size and T is current sequence length
            - log probablity of candidate tokens
              :math:`(N, B * B * 2)` where N is batch size and B is beam size.
        """
        seqlen = tokens.size(-1)
        logits = self._decoder(tokens.view(-1, seqlen), memory,
                               memory_padding_mask=memory_padding_mask)[:, -1, :]
        logits = F.log_softmax(logits, dim=-1)
        next_token_score, next_token = logits.topk(self._beam * 2, dim=-1)
        next_token_score = next_token_score.view(-1, self._beam, self._beam * 2)
        scores = scores.unsqueeze(dim=2) + next_token_score
        scores = scores.view(-1, self._beam ** 2 * 2)
        tokens = torch.cat([tokens.unsqueeze(dim=2).repeat(1, 1, self._beam * 2, 1),
                            next_token.view(-1, self._beam, self._beam * 2, 1)],
                           dim=-1)
        tokens = tokens.view(-1, self._beam ** 2 * 2, seqlen + 1)
        return tokens, scores

    def _add_finished(self, finished, finished_scores, tokens, scores, finished_mask):
        """
        Select new finished sequences and add them to finished list.

        Args:
            finished: finished sequences ending with eos
                :math:`(N, S, maxT)` where N is batch size, maxT is the maximum sequence length
                    S is the number of return sequence
            finished_scores: log probability of finished sequence
                :math:`(N, S)` where N is batch size and B is beam size
                    S is the number of return sequence
            tokens: newly-generated sequences
                :math:`(N, B * B * 2, T)` where N is batch size, B is beam size and T is decoded sequence length
            scores: log probability of newly-generated sequences
                :math:`(N, B)` where N is batch size, B is beam size.
            finished_mask: mask indicating which sequences in `tokens` ends with eos
                :math:`(N, B)` where N is batch size and B is beam size

        Returns:
            - updated finished sequences
                :math:`(N, S, maxT)` where N is batch size, B is beam size, maxT is the maximum sequence length
                    S is the number of return sequence
            - log probability of updated finished sequence
                :math:`(N, S)` where N is batch size and B is beam size.
                    S is the number of return sequence
        """
        maxlen = finished.size(dim=-1)
        curlen = tokens.size(dim=-1)
        non_finished_mask = ~finished_mask
        scores = scores / ((5 + curlen) / 6)**self._lenpen
        scores = scores.masked_fill(non_finished_mask, float('-inf'))
        tokens = torch.cat([tokens, tokens.data.new(tokens.size()[:-1] + ((maxlen - curlen),)).fill_(self._eos)],
                           dim=-1)
        finished_scores = torch.cat([finished_scores, scores], dim=1)
        finished = torch.cat([finished, tokens], dim=1)
        finished_scores, idx = finished_scores.topk(self._num_return_sequence, dim=1)
        finished = finished.gather(1, index=idx.unsqueeze(dim=-1).repeat(1, 1, maxlen))
        return finished, finished_scores

    def _update_states(self, tokens, scores, finished_mask):
        """
        Update decoder internal states with results preferred by beam search algorithm

        Args:
            tokens: newly-generated sequences
                :math:`(N, B * B * 2, T)` where N is batch size, B is beam size and T is sequence length
            scores: log probability of newly-generated sequences
                :math:`(N, B * B * 2)` where N is batch size, B is beam size.
            finished_mask: mask indicating which sequences in `tokens` ends with eos
                :math:`(N, B * B * 2)` where N is batch size and B is beam size

        Returns:
            - updated sequences
                :math: `(N, B, T)` where N is batch size, B is beam size and T is sequence length
            - scores of updated sequences
                :math: `(N, B)` where N is batch size, B is beam size
        """
        curlen = tokens.size(dim=-1)
        scores = scores.masked_fill(finished_mask, float('-inf'))
        scores, idx = scores.topk(self._beam, dim=-1)

        tokens = tokens.gather(1, index=idx.unsqueeze(2).repeat(1, 1, curlen))
        cache = self._decoder.get_cache()
        cache = self._update_cache(cache, idx)
        self._decoder.set_cache(cache)
        return tokens, scores

    def _expand(self, memory, memory_padding_mask):
        """
        Expand encoder states with `beam` times

        Args:
            memory: memory for attention.
              :math:`(M, N, E)`, where M is the memory sequence length, N is the batch size,
              B is beam size and E is feature size
            memory_padding_mask: memory sequence padding mask.
              :math:`(N, M)` where M is the memory sequence length, N is the batch size,
              B is beam size.

        Returns:
            - expanded memory
              :math:`(M, N * B, E)`, where M is the memory sequence length, N is the batch size,
              B is beam size and E is feature size
            - expanded memory padding mask
              :math:`(N * B, M)` where M is the memory sequence length, N is the batch size,
              B is beam size.
        """
        batch_size, memory_size = memory_padding_mask.size()
        cache = self._decoder.get_cache()
        cache = self._expand_cache(cache)
        self._decoder.set_cache(cache)
        memory = memory.unsqueeze(dim=2).repeat(1, 1, self._beam, 1)
        memory = memory.view(memory_size, batch_size * self._beam, -1)
        memory_padding_mask = memory_padding_mask.unsqueeze(dim=1).repeat(1, self._beam, 1)
        memory_padding_mask = memory_padding_mask.view(batch_size * self._beam, memory_size)
        return memory, memory_padding_mask

    def _expand_tensor(self, cache: torch.Tensor):
        """
        Expand tensor with `beam` times after batch dimension

        Args:
            cache: torch tensor
                :math:`(M, N, *)` where M is the memory sequence length, N is the batch size.

        Returns:
            - expanded cache
                :math:`(M, N * B, *)` where M is the memory sequence length, N is the batch size,
              B is beam size.
        """
        size = cache.size()
        repeat = (1, 1, self._beam) + tuple(1 for _ in size[2:])
        cache = cache.unsqueeze(2).repeat(repeat).view((size[0], size[1] * self._beam,) + size[2:])
        return cache

    def _update_tensor(self, cache: torch.Tensor, idx):
        """
        Update tensor by indexing with idx

        Args:
            cache: torch tensor
                :math:`(M, N * B * B, E)` where M is the memory sequence length, N is the batch size,
                B is beam size, and E is feature dimension.
            idx: index to select
                :math:`(N, B)` where N is the batch size and B is beam size.

        Returns:
            - indexed cache
                :math:`(M, N * B, E)` where N is the batch size, B is beam size and E is feature dimension.
        """
        size = cache.size()
        idx = idx // (self._beam * 2)
        cache = cache.view(size[0], size[1] // self._beam, self._beam, size[-1])
        idx = idx.unsqueeze(0).unsqueeze(-1).repeat((size[0], 1, 1, size[-1]))
        cache = cache.gather(2, index=idx)
        cache = cache.view(size[0], size[1], size[-1])
        return cache


@register_search
class BeamSearchV2(SequenceSearch):
    """
    BeamSearchV2 is beam search on sequence generation. Different from V1, V2 is another implementation of beam search,
    reducing beam search candidates at next step from beam * beam * 2 to beam * 2.

    Args:
        beam: beam size
        lenpen: length penalty
        maxlen_coef (a, b): maxlen computation coefficient.
            The max length is computed as `(S * a + b)`, where S is source sequence length.
        maxlen: maximum length
        num_return_sequence: the number of return generated sequence for each input sample,
            which must smaller or equal to beam size, default as 1 to return the sequence
            with the highest log probability
        keepdim: bool, squeeze the output sequence if set True
    """

    def __init__(self, beam=4, lenpen=0.1, maxlen_coef=(1.2, 10), maxlen=1000, num_return_sequence=1, keepdim=False):
        super().__init__()
        self._beam = beam
        self._lenpen = lenpen
        self._maxlen = maxlen
        self._maxlen_a, self._maxlen_b = maxlen_coef
        self._num_return_sequence = num_return_sequence
        self._keepdim = keepdim
        self._update_cache = recursive(self._update_tensor)

        assert self._num_return_sequence <= self._beam, \
            f"number of return sequence <= beam size, got {self.num_return_sequence} > {self._beam}"

    def forward(self,
                tokens,
                memory,
                memory_padding_mask,
                target_mask=None,
                prev_scores=None):
        """
        Decoding full-step sequence with beam search

        Args:
            tokens: previous tokens or prefix of sequence
              :math:`(N, T)` where N is batch size and T is current sequence length
            memory: memory for attention.
              :math:`(M, N, E)`, where M is the memory sequence length, N is the batch size,
            memory_padding_mask: memory sequence padding mask.
              :math:`(N, M)` where M is the memory sequence length, N is the batch size.
            target_mask: target mask indicating blacklist tokens
              :math:`(N, V)` where N is batch size and V is vocab size
            prev_scores: scores of previous tokens
              :math:`(N)` where N is batch size

        Returns:
            - log probability of generated sequence
                :math: `(N, S)` where N is batch size and S is the number of return sequence
            - generated sequence
                :math: `(N, S, maxT)` where N is batch size, maxT is the maximum sequence length
                    and S is the number of return sequence
                       `(N, maxT)` if self.keepdim=True and S = 1
        """
        batch_size, curlen = tokens.size(0), tokens.size(1)
        memory_size = memory_padding_mask.size(-1)

        # copy token and memory for 'beam' times
        tokens = self._expand(tokens, dim=0)  # [N * B, T]
        memory = self._expand(memory, dim=1)  # [M, N * B, E]
        memory_padding_mask = self._expand(memory_padding_mask, dim=0)  # [N * B, M]
        if prev_scores is None:
            scores = create_init_scores(tokens, memory)  # [N * B]
        else:
            scores = self._expand(prev_scores, dim=1)  # [N * B]

        # generate rest of tokens
        maxlen = min(int(memory_size * self._maxlen_a + self._maxlen_b) - curlen, self._maxlen)
        finished = tokens.data.new(batch_size, 1, maxlen + curlen + 1).fill_(self._eos)
        finished_scores = scores.data.new(batch_size, 1).fill_(float('-inf'))
        for step in range(maxlen - 1):
            tokens, scores, idx = self._produce_candidates(tokens, memory, memory_padding_mask, scores, step)
            finished_mask = tokens[:, :, -1] == self._eos
            finished, finished_scores = self._add_finished(finished, finished_scores, tokens, scores, finished_mask)
            tokens, scores = self._update_states(tokens, scores, idx, finished_mask)

        if not self._keepdim and self._num_return_sequence == 1:
            finished = finished.squeeze(dim=1)
        return finished_scores, finished

    def _produce_candidates(self, tokens, memory, memory_padding_mask, scores, step):
        """
        Predict `beam * 2` candidates at next step

        Args:
            tokens: previous tokens or prefix of sequence at time step T
              :math:`(N * B, T)` where N is batch size, B is beam size and T is current sequence length
            memory: memory for attention.
              :math:`(M, N * B, E)`, where M is the memory sequence length, N is the batch size,
              B is beam size and E is feature size
            memory_padding_mask: memory sequence padding mask.
              :math:`(N * B, M)` where M is the memory sequence length, N is the batch size,
              B is beam size.
            scores: scores of previous tokens
              :math:`(N * B)` where N is batch size and B is beam size.
            step: decoding steps

        Returns:
            - candidate tokens at time step T+1
              :math:`(N, B * B * 2, T+1)` where N is batch size, B is beam size and T is current sequence length
            - log probablity of candidate tokens
              :math:`(N, B * B * 2)` where N is batch size and B is beam size.
        """
        batch_size, seqlen = tokens.size(0) // self._beam, tokens.size(-1)
        logits = self._decoder(tokens, memory,
                               memory_padding_mask=memory_padding_mask)[:, -1, :]
        logits = F.log_softmax(logits, dim=-1)
        vocab_size = logits.size(-1)
        tokens = tokens.view(batch_size, self._beam, seqlen)
        scores = scores.unsqueeze(dim=-1) + logits

        scores = scores.view(batch_size, -1)
        if step == 0:
            scores = scores[:, :vocab_size]
        scores, next_id = scores.topk(self._beam * 2, dim=-1)
        next_token_batch_id, next_token = next_id // vocab_size, next_id % vocab_size

        prev_tokens = tokens.gather(1, next_token_batch_id.unsqueeze(dim=-1).repeat(1, 1, seqlen))
        tokens = torch.cat([prev_tokens, next_token.unsqueeze(dim=-1)], dim=-1)
        return tokens, scores, next_token_batch_id

    def _add_finished(self, finished, finished_scores, tokens, scores, finished_mask):
        """
        Select new finished sequences and add them to finished list.

        Args:
            finished: finished sequences ending with eos
                :math:`(N, S, maxT)` where N is batch size, maxT is the maximum sequence length
                    S is the number of return sequence
            finished_scores: log probability of finished sequence
                :math:`(N, S)` where N is batch size and B is beam size
                    S is the number of return sequence
            tokens: newly-generated sequences
                :math:`(N, B * B * 2, T)` where N is batch size, B is beam size and T is decoded sequence length
            scores: log probability of newly-generated sequences
                :math:`(N, B)` where N is batch size, B is beam size.
            finished_mask: mask indicating which sequences in `tokens` ends with eos
                :math:`(N, B)` where N is batch size and B is beam size

        Returns:
            - updated finished sequences
                :math:`(N, S, maxT)` where N is batch size, B is beam size, maxT is the maximum sequence length
                    S is the number of return sequence
            - log probability of updated finished sequence
                :math:`(N, S)` where N is batch size and B is beam size.
                    S is the number of return sequence
        """
        maxlen = finished.size(dim=-1)
        curlen = tokens.size(dim=-1)
        non_finished_mask = ~finished_mask
        scores = scores / ((5 + curlen) / 6) ** self._lenpen
        scores = scores.masked_fill(non_finished_mask, float('-inf'))
        tokens = torch.cat([tokens, tokens.data.new(tokens.size()[:-1] + ((maxlen - curlen),)).fill_(self._eos)],
                           dim=-1)
        finished_scores = torch.cat([finished_scores, scores], dim=1)
        finished = torch.cat([finished, tokens], dim=1)
        finished_scores, idx = finished_scores.topk(self._num_return_sequence, dim=1)
        finished = finished.gather(1, index=idx.unsqueeze(dim=-1).repeat(1, 1, maxlen))
        return finished, finished_scores

    def _update_states(self, tokens, scores, index, finished_mask):
        """
        Update decoder internal states with results preferred by beam search algorithm

        Args:
            tokens: newly-generated sequences
                :math:`(N, B * 2, T)` where N is batch size, B is beam size and T is sequence length
            scores: log probability of newly-generated sequences
                :math:`(N, B * 2)` where N is batch size, B is beam size.
            index: selected beam indices
                :math:`(N, B * 2)` where N is batch size, B is beam size
            finished_mask: mask indicating which sequences in `tokens` ends with eos
                :math:`(N, B * 2)` where N is batch size and B is beam size

        Returns:
            - updated sequences
                :math: `(N, B, T)` where N is batch size, B is beam size and T is sequence length
            - scores of updated sequences
                :math: `(N, B)` where N is batch size, B is beam size
        """
        curlen = tokens.size(dim=-1)
        scores = scores.masked_fill(finished_mask, float('-inf'))
        scores, idx = scores.topk(self._beam, dim=-1)
        tokens = tokens.gather(1, index=idx.unsqueeze(2).repeat(1, 1, curlen))

        cache = self._decoder.get_cache()
        cache = self._update_cache(cache, index, idx)
        self._decoder.set_cache(cache)
        return tokens.view(-1, curlen), scores.view(-1)

    def _expand(self, tensor, dim):
        """
        Expand encoder states with `beam` times

        Args:
            tensor: a torch tensor


        Returns:
            - expanded tensor
        """
        tensor_shape = tensor.size()
        tensor = tensor.unsqueeze(dim=dim+1).repeat(
            (1,) * (dim+1) + (self._beam, ) + (1, ) * (len(tensor_shape) - (dim+1)))
        tensor = tensor.view(
            tuple(tensor_shape[:dim]) + (tensor_shape[dim] * self._beam,) + tuple(tensor_shape[dim+1:]))
        return tensor

    def _update_tensor(self, cache: torch.Tensor, beam_index, non_finished_index):
        """
        Update tensor by indexing with idx

        Args:
            cache: torch tensor
                :math:`(M, N * B * B, E)` where M is the memory sequence length, N is the batch size,
                B is beam size, and E is feature dimension.
            beam_index: index to select in top B * 2
                :math:`(N, B)` where N is the batch size and B is beam size.
            non_finished_index: index to select in top B
                :math:`(N, B)` where N is the batch size and B is beam size.

        Returns:
            - indexed cache
                :math:`(M, N * B, E)` where N is the batch size, B is beam size and E is feature dimension.
        """
        size = cache.size()
        cache = cache.view(size[0], size[1] // self._beam, self._beam, size[-1])

        idx = beam_index.unsqueeze(0).unsqueeze(-1).repeat((size[0], 1, 1, size[-1]))
        cache = cache.gather(2, index=idx)

        idx = non_finished_index // (self._beam * 2)
        idx = idx.unsqueeze(0).unsqueeze(-1).repeat((size[0], 1, 1, size[-1]))
        cache = cache.gather(2, index=idx)

        cache = cache.view(size[0], size[1], size[-1])
        return cache


@register_search
class BeamSearchV3(SequenceSearch):
    """
    BeamSearchV3 is beam search on sequence generation. Different from V1 and V2, V3 use another strategy to implement
    beam search by ranking finished and unfinished candidates all together.

    Args:
        beam: beam size
        lenpen: length penalty
        maxlen_coef (a, b): maxlen computation coefficient.
            The max length is computed as `(S * a + b)`, where S is source sequence length.
        maxlen: maximum length
        num_return_sequence: the number of return generated sequence for each input sample,
            which must smaller or equal to beam size, default as 1 to return the sequence
            with the highest log probability
        keepdim: bool, squeeze the output sequence if set True
    """

    def __init__(self, beam=4, lenpen=0.1, maxlen_coef=(1.2, 10), maxlen=1000, num_return_sequence=1, keepdim=False):
        super().__init__()
        self._beam = beam
        self._lenpen = lenpen
        self._maxlen = maxlen
        self._maxlen_a, self._maxlen_b = maxlen_coef
        self._num_return_sequence = num_return_sequence
        self._keepdim = keepdim
        self._update_cache = recursive(self._update_tensor)

        assert self._num_return_sequence <= self._beam, \
            f"number of return sequence <= beam size, got {self.num_return_sequence} > {self._beam}"

    def forward(self,
                tokens,
                memory,
                memory_padding_mask,
                target_mask=None,
                prev_scores=None):
        """
        Decoding full-step sequence with beam search

        Args:
            tokens: previous tokens or prefix of sequence
              :math:`(N, T)` where N is batch size and T is current sequence length
            memory: memory for attention.
              :math:`(M, N, E)`, where M is the memory sequence length, N is the batch size,
            memory_padding_mask: memory sequence padding mask.
              :math:`(N, M)` where M is the memory sequence length, N is the batch size.
            target_mask: target mask indicating blacklist tokens
              :math:`(N, V)` where N is batch size and V is vocab size
            prev_scores: scores of previous tokens
              :math:`(N)` where N is batch size

        Returns:
            - log probability of generated sequence
                :math: `(N, S)` where N is batch size and S is the number of return sequence
            - generated sequence
                :math: `(N, S, maxT)` where N is batch size, maxT is the maximum sequence length
                    and S is the number of return sequence
                       `(N, maxT)` if self.keepdim=True and S = 1
        """
        batch_size, curlen = tokens.size(0), tokens.size(1)
        memory_size = memory_padding_mask.size(-1)

        # copy token and memory for 'beam' times
        tokens = self._expand(tokens, dim=0)  # [N * B, T]
        memory = self._expand(memory, dim=1)  # [M, N * B, E]
        memory_padding_mask = self._expand(memory_padding_mask, dim=0)  # [N * B, M]
        if prev_scores is None:
            scores = create_init_scores(tokens, memory)  # [N * B]
        else:
            scores = self._expand(prev_scores, dim=1)  # [N * B]

        # generate rest of tokens
        maxlen = min(int(memory_size * self._maxlen_a + self._maxlen_b) - curlen, self._maxlen)
        for step in range(maxlen - 1):
            tokens, scores, idx = self._produce_candidates(tokens, memory, memory_padding_mask, scores, step)
            self._update_states(idx)

        scores = scores.view(batch_size, self._beam)[:, :self._num_return_sequence]
        tokens = tokens.view(batch_size, self._beam, -1)[:, :self._num_return_sequence, :]

        if not self._keepdim and self._num_return_sequence == 1:
            tokens = tokens.squeeze(dim=1)
        return scores, tokens

    def _produce_candidates(self, tokens, memory, memory_padding_mask, scores, step):
        """
        Predict `beam` candidates at next step

        Args:
            tokens: previous tokens or prefix of sequence at time step T
              :math:`(N * B, T)` where N is batch size, B is beam size and T is current sequence length
            memory: memory for attention.
              :math:`(M, N * B, E)`, where M is the memory sequence length, N is the batch size,
              B is beam size and E is feature size
            memory_padding_mask: memory sequence padding mask.
              :math:`(N * B, M)` where M is the memory sequence length, N is the batch size,
              B is beam size.
            scores: scores of previous tokens
              :math:`(N * B)` where N is batch size and B is beam size.
            step: decoding steps

        Returns:
            - candidate tokens at time step T+1
              :math:`(N, B * B, T+1)` where N is batch size, B is beam size and T is current sequence length
            - log probability of candidate tokens
              :math:`(N, B * B)` where N is batch size and B is beam size.
        """
        batch_size, seqlen = tokens.size(0) // self._beam, tokens.size(-1)
        logits = self._decoder(tokens, memory,
                               memory_padding_mask=memory_padding_mask)[:, -1, :]
        logits = F.log_softmax(logits, dim=-1)
        finished = tokens[:, -1].eq(self._eos).type_as(logits)
        logits[:, self._eos] *= 1 - finished

        vocab_size = logits.size(-1)
        tokens = tokens.view(batch_size, self._beam, seqlen)
        scores = scores.unsqueeze(dim=-1) + logits
        scores = scores.view(batch_size, self._beam, -1)

        normalized_scores = self._normalize(tokens, scores)
        normalized_scores = normalized_scores.view(batch_size, -1)
        if step == 0:
            normalized_scores = normalized_scores[:, :vocab_size]
        _, next_id = normalized_scores.topk(self._beam, dim=-1)
        next_token_batch_id, next_token = next_id // vocab_size, next_id % vocab_size

        scores = scores.view(batch_size, -1)
        scores = scores.gather(1, next_id)

        prev_tokens = tokens.gather(1, next_token_batch_id.unsqueeze(dim=-1).repeat(1, 1, seqlen))
        tokens = torch.cat([prev_tokens, next_token.unsqueeze(dim=-1)], dim=-1)
        return tokens.view(-1, seqlen + 1), scores.view(-1), next_token_batch_id

    def _normalize(self, tokens, scores):
        """
        Select new finished sequences and add them to finished list.

        Args:
            tokens: newly-generated sequences
                :math:`(N, B * B, T)` where N is batch size, B is beam size and T is decoded sequence length
            scores: log probability of newly-generated sequences
                :math:`(N, B)` where N is batch size, B is beam size.

        Returns:
            - updated finished sequences
                :math:`(N, S, maxT)` where N is batch size, B is beam size, maxT is the maximum sequence length
                    S is the number of return sequence
            - log probability of updated finished sequence
                :math:`(N, S)` where N is batch size and B is beam size.
                    S is the number of return sequence
        """
        length = tokens.ne(self._eos).type_as(scores).sum(dim=-1, keepdim=True) - 1
        scores = scores / ((5 + length) / 6) ** self._lenpen
        return scores

    def _update_states(self, index):
        """
        Update decoder internal states with results preferred by beam search algorithm

        Args:
            index: selected beam indices
                :math:`(N, B)` where N is batch size, B is beam size

        Returns:
            - updated sequences
                :math: `(N, B, T)` where N is batch size, B is beam size and T is sequence length
            - scores of updated sequences
                :math: `(N, B)` where N is batch size, B is beam size
        """
        cache = self._decoder.get_cache()
        cache = self._update_cache(cache, index)
        self._decoder.set_cache(cache)

    def _expand(self, tensor, dim):
        """
        Expand encoder states with `beam` times

        Args:
            tensor: a torch tensor


        Returns:
            - expanded tensor
        """
        tensor_shape = tensor.size()
        tensor = tensor.unsqueeze(dim=dim+1).repeat(
            (1,) * (dim+1) + (self._beam, ) + (1, ) * (len(tensor_shape) - (dim+1)))
        tensor = tensor.view(
            tuple(tensor_shape[:dim]) + (tensor_shape[dim] * self._beam,) + tuple(tensor_shape[dim+1:]))
        return tensor

    def _update_tensor(self, cache: torch.Tensor, index):
        """
        Update tensor by indexing with idx

        Args:
            cache: torch tensor
                :math:`(M, N * B * B, E)` where M is the memory sequence length, N is the batch size,
                B is beam size, and E is feature dimension.
            index: index to select in top B
                :math:`(N, B)` where N is the batch size and B is beam size.

        Returns:
            - indexed cache
                :math:`(M, N * B, E)` where N is the batch size, B is beam size and E is feature dimension.
        """
        size = cache.size()
        cache = cache.view(size[0], size[1] // self._beam, self._beam, size[-1])

        idx = index.unsqueeze(0).unsqueeze(-1).repeat((size[0], 1, 1, size[-1]))
        cache = cache.gather(2, index=idx)

        cache = cache.view(size[0], size[1], size[-1])
        return cache
