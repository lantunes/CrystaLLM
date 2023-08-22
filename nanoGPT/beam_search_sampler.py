import math
import traceback
from typing import Tuple, List, Iterable, Callable

import torch

from model import GPT, GPTConfig
from lib import (
    CIFTokenizer,
    CIFScorer,
    bond_length_reasonableness_score,
    is_formula_consistent,
    is_space_group_consistent,
    is_atom_site_multiplicity_consistent,
    extract_space_group_symbol,
    replace_symmetry_operators,
    remove_atom_props_block
)


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

    def is_complete(self, newline_id: int, min_len: int = 90) -> bool:
        """
        Returns True if this hypothesis is complete, and False otherwise.

        :param newline_id: the token ID for `\n`
        :param min_len: the minimum length a sequence should have
        :return: True if the hypothesis is considered complete, and False otherwise
        """
        return len(self.token_sequence) >= min_len and self.token_sequence[-2:] == [newline_id, newline_id]


class BeamSearchSampler:
    """
    A sampler that uses beam search to sample the most probable CIF
    from a trained language model.
    """
    def __init__(self, model: GPT, config: GPTConfig, tokenizer: CIFTokenizer,
                 scorer: CIFScorer = None, k=1, temperature=1.0,
                 bond_length_acceptability_cutoff=1.0, score_multiplier=-1) -> None:
        """
        Construct an instance of a `BeamSearchSampler`.

        :param model: the language model to use during search
        :param config: the GPT config
        :param tokenizer: the CIF tokenizer to use
        :param scorer: an external scorer for scoring completed CIFs
        :param k: the beam size to use during search
        :param temperature: the temperature to use for scaling the logits
        :param bond_length_acceptability_cutoff: the bond length acceptability cutoff
        :param score_multiplier: a factor with which to multiply the score (i.e. 1, or -1)
        """
        self._model = model
        self._model.eval()
        self._config = config
        self._tokenizer = tokenizer
        self._scorer = scorer
        self._k = k
        self._temperature = temperature
        self._bond_length_acceptability_cutoff = bond_length_acceptability_cutoff
        self._score_multiplier = score_multiplier

    def _postprocess(self, cif_str):
        # replace the symmetry operators with the correct operators
        space_group_symbol = extract_space_group_symbol(cif_str)
        if space_group_symbol is not None and space_group_symbol != "P 1":
            cif_str = replace_symmetry_operators(cif_str, space_group_symbol)

        # remove atom props
        cif_str = remove_atom_props_block(cif_str)

        return cif_str

    def _is_valid(self, generated_cif):
        if not is_formula_consistent(generated_cif):
            msg = "the generated CIF is inconsistent in terms of composition"
            return False, msg

        if not is_atom_site_multiplicity_consistent(generated_cif):
            msg = "the generated CIF is inconsistent in terms of atom site multiplicity"
            return False, msg

        bond_length_score = bond_length_reasonableness_score(generated_cif)
        if bond_length_score < self._bond_length_acceptability_cutoff:
            msg = f"unreasonable bond lengths detected " \
                  f"({(1-bond_length_score)*100:.0f}% of bond lengths were found to be unreasonable)"
            return False, msg

        if not is_space_group_consistent(generated_cif):
            msg = "the generated CIF is inconsistent in terms of space group"
            return False, msg

        return True, ""

    def sample(self, start: str, min_len: int, device: str) -> Tuple[str, float, float]:
        """
        Sample a CIF from the model, and computes its probability,
        using a beam search strategy.

        :param start: the starting prompt
        :param min_len: the minimum length the sequence should have
        :param device: the device the model has been transferred to
        :return: a CIF
        :return: the log probability of the CIF under the trained model
        :return: the score (the same as the log prob. if no scorer was specified)
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
                    if hypothesis.is_complete(newline_id, min_len=min_len):
                        log_prob = hypothesis.log_prob
                        cif = decode(hypothesis.token_sequence)

                        try:
                            cif = self._postprocess(cif)
                            valid, msg = self._is_valid(cif)
                            if not valid:
                                print(f"CIF invalid: {msg}")
                                continue
                        except Exception as e:
                            print(f"exception while post-processing and validating: {e}")
                            print(traceback.format_exc())
                            continue

                        # optionally score the completed hypothesis with an external scorer
                        if self._scorer is not None:
                            try:
                                print("invoking external scorer...")
                                score = self._score_multiplier * self._scorer.score(cif)
                                print(f"external scorer returned score: {score}")
                            except Exception as e:
                                print(f"exception while using external scorer: {e}")
                                print(traceback.format_exc())
                                continue
                        else:
                            score = log_prob

                        completed.append((cif, log_prob, score))
                    else:
                        beam.append(hypothesis)

        if len(completed) > 0:
            best_cif, best_log_prob, best_score = self._sort_descending(completed, key=lambda comp: comp[2])[0]
            print(f"best log prob: {best_log_prob:.5f}; best score: {best_score:.5f}")
            return best_cif, best_log_prob, best_score

        return "", math.nan, math.nan

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

        # we will always keep the best hypotheses according to log prob, since only completed
        #  hypotheses can be scored with an external scorer (e.g. we can't assess formation
        #  energy of an incomplete CIF)
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
