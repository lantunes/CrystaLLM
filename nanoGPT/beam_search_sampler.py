import math
from typing import Tuple, List, Iterable, Callable

import torch

from model import GPT, GPTConfig
from lib import get_cif_tokenizer, CIFTokenizer


class Hypothesis:
    """
    Representation of a hypothesis that is maintained during beam search.
    """
    def __init__(self, token_sequence: List[int], log_prob: float) -> None:
        """
        Construct an instance of a `Hypothesis`.

        :param token_sequence: a sequence of tokens representing a tokenized CIF
        :param log_prob: the log probability of the sequence
        """
        self.token_sequence = token_sequence
        self.log_prob = log_prob

    def is_complete(self, newline_id: int) -> bool:
        """
        Returns True if this hypothesis is complete, and False otherwise.

        :param newline_id: the token ID for `\n`
        :return: True if the hypothesis is considered complete, and False otherwise
        """
        return self.token_sequence[-2:] == [newline_id, newline_id]


class BeamSearchSampler:
    """
    A sampler that uses beam search to sample the most probable CIF
    from a trained language model.
    """
    def __init__(self, model: GPT, config: GPTConfig, tokenizer: CIFTokenizer, k=1, temperature=1.0) -> None:
        """
        Construct an instance of a `BeamSearchSampler`.

        :param model: the language model to use during search
        :param config: the GPT config
        :param tokenizer: the CIF tokenizer to use
        :param k: the beam size to use during search
        :param temperature: the temperature to use for scaling the logits
        """
        self._model = model
        self._config = config
        self._tokenizer = tokenizer
        self._model.eval()
        self._k = k
        self._temperature = temperature

    def sample(self, start: str, device: str) -> Tuple[str, float]:
        """
        Sample a CIF from the model, and computes its probability,
        using a beam search strategy.

        :param start: the starting prompt
        :param device: the device the model has been transferred to
        :return: a CIF
        :return: the probability of the CIF under the trained model
        """
        print("sampling...")

        encode = self._tokenizer.encode
        decode = self._tokenizer.decode
        newline_id = self._tokenizer.token_to_id["\n"]

        start_ids = encode(self._tokenizer.tokenize_cif(start))

        # the completed hypotheses, one of which we'll consider returning
        completed = []

        with torch.no_grad():
            child_ids = list(range(len(self._tokenizer.token_to_id)))

            # initialize the beam
            beam = [Hypothesis(start_ids, 0)]

            while len(beam) > 0:
                # advance the beam
                advanced_beam = self._advance_beam(beam, child_ids, self._k, device)

                beam = []
                # remove completed hypotheses
                for hypothesis in advanced_beam:
                    if hypothesis.is_complete(newline_id):
                        completed.append((hypothesis.token_sequence, hypothesis.log_prob))
                    else:
                        beam.append(hypothesis)

        best_hypothesis, best_hypothesis_log_prob = self._sort_descending(completed, key=lambda pair: pair[1])[0]
        cif = decode(best_hypothesis)
        prob = math.exp(best_hypothesis_log_prob)

        print(f"best log prob: {best_hypothesis_log_prob:.5f}")

        return cif, prob

    def _advance_beam(self, beam: List[Hypothesis], child_tokens: List[int], k: int, device: str) -> List[Hypothesis]:
        """
        Advance the given beam by extending all its hypotheses by a single child token.

        :param beam: the beam to advance
        :param child_tokens: the list of child tokens to use
        :param k: the beam size
        :param device: the device to use
        :return: an advanced beam
        """
        new_beam = []
        for hypothesis in beam:
            idx = (torch.tensor(hypothesis.token_sequence, dtype=torch.long, device=device)[None, ...])

            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self._config.block_size else idx[:, -self._config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self._model(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / self._temperature

            dist = torch.distributions.categorical.Categorical(logits=logits)

            for child_token in child_tokens:
                new_token_sequence = list(hypothesis.token_sequence)
                new_token_sequence.append(child_token)
                log_prob = dist.log_prob(torch.tensor(child_token, device=device).view(1)).item()
                new_beam.append(Hypothesis(new_token_sequence, hypothesis.log_prob + log_prob))

        scored_new_beam = self._sort_descending(new_beam, key=lambda pair: pair.log_prob)[:k]

        return scored_new_beam

    @staticmethod
    def _sort_descending(sequence: Iterable, key: Callable) -> List:
        """
        Sort the given sequence in descending order, using the given callable
        to customize the sort order.

        :param sequence: a sequence to sort
        :param key: a callable to use during the sort
        :return: the sorted sequence as a list
        """
        return list(reversed(sorted(sequence, key=key)))
