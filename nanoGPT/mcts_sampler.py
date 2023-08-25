import os
import random
import math
from math import sqrt
import traceback
from typing import List

import numpy as np
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


def is_sequence_complete(token_sequence, newline_id):
    return len(token_sequence) > 1 and token_sequence[-2:] == [newline_id, newline_id]


class MCTSEvaluator:
    def __init__(self, scorer: CIFScorer, tokenizer: CIFTokenizer,
                 bond_length_acceptability_cutoff=1.0, reward_k=2.0, out_dir=None):
        self._scorer = scorer
        self._tokenizer = tokenizer
        self._bond_length_acceptability_cutoff = bond_length_acceptability_cutoff
        self._k = reward_k
        self._out_dir = out_dir
        self._num_valid = 0
        self._all_scores = []
        self._all_cifs = []

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
            return False, msg, None

        if not is_atom_site_multiplicity_consistent(generated_cif):
            msg = "the generated CIF is inconsistent in terms of atom site multiplicity"
            return False, msg, None

        bond_length_score = bond_length_reasonableness_score(generated_cif)
        if bond_length_score < self._bond_length_acceptability_cutoff:
            msg = f"unreasonable bond lengths detected " \
                  f"({(1 - bond_length_score) * 100:.0f}% of bond lengths were found to be unreasonable)"
            return False, msg, bond_length_score

        if not is_space_group_consistent(generated_cif):
            msg = "the generated CIF is inconsistent in terms of space group"
            return False, msg, None

        return True, "", None

    def _get_reward(self, score):
        """
        Returns a number between 0 and 1 representing the reward.
        Because we don't know the scale of the scores beforehand, we'll
        keep a running average of scores observed. A score which is close
        to the mean observed score will result in a reward close to 0.5.
        The k value determines how sensitive the reward is to deviations
        from the mean score.
        If higher scores are better, then provide a negative k value. If
        lower scores are better, provide a positive k value.
        """
        self._all_scores.append(score)
        if len(self._all_scores) == 1:
            # when we only have a single sample, the reward should be 0.5
            return 0.5
        mu = np.mean(self._all_scores)
        sigma = np.std(self._all_scores)
        return 1 / (1 + math.e**(self._k*((score - mu)/sigma)))

    def _write_cif_to_file(self, cif, score, reward, id):
        if self._out_dir is not None:

            if not os.path.exists(self._out_dir):
                os.makedirs(self._out_dir)

            cif_file = f"generated_{id}.cif"
            cif_fname = os.path.join(self._out_dir, cif_file)
            if not os.path.exists(cif_fname):
                print(f"writing CIF to: {cif_fname}")
                with open(cif_fname, "wt") as f:
                    f.write(cif)

                # create .csv to keep track of results
                csv_file = "results.csv"
                csv_fname = os.path.join(self._out_dir, csv_file)
                if not os.path.exists(csv_fname):
                    print(f"creating {csv_fname} as it does not exist...")
                    with open(csv_fname, "wt") as f:
                        f.write("file,score,reward\n")

                # update .csv
                with open(csv_fname, "a") as f:
                    f.write(f"{cif_file},{score},{reward}\n")

            else:
                print(f"CIF not written to file as it already exists: {cif_fname}")

    def __call__(self, token_sequence):
        cif = self._tokenizer.decode(token_sequence)

        try:
            cif = self._postprocess(cif)
            valid, msg, bond_length_score = self._is_valid(cif)
            if not valid:
                print(f"CIF invalid: {msg}")
                if bond_length_score is not None:
                    return -(1 - bond_length_score)
                else:
                    return -1.0
        except Exception as e:
            print(f"exception while post-processing and validating: {e}")
            print(traceback.format_exc())
            return -1.0

        self._num_valid += 1

        try:
            print("invoking external scorer...")
            score = self._scorer.score(cif)
            print(f"external scorer returned score: {score}")
        except Exception as e:
            print(f"exception while scoring: {e}")
            print(traceback.format_exc())
            return -1.0

        if math.isnan(score):
            print(f"reward cannot be computed as score is nan")
            return -1.0

        reward = self._get_reward(score)
        print(f"computed reward: {reward}")

        self._write_cif_to_file(cif, score, reward, self._num_valid)

        return reward


class MCTSSampler:

    def __init__(self, model: GPT, config: GPTConfig, width, max_depth, eval_function, cpuct,
                 tokenizer: CIFTokenizer, temperature: float, device: str):
        self._width = width
        self._max_depth = max_depth
        self._eval_function = eval_function
        self._best_sequence = None
        self._cpuct = cpuct
        self._tokenizer = tokenizer
        child_ids = list(range(len(self._tokenizer.token_to_id)))
        self._lm = _LanguageModel(model, config, child_ids=child_ids, temperature=temperature, device=device)
        self._newline_id = self._tokenizer.token_to_id["\n"]

    def search(self, start: str, num_simulations: int):
        state = self._tokenizer.encode(self._tokenizer.tokenize_cif(start))
        root_node = _Node(state, self._lm, self._width, self._max_depth, self._cpuct, self._newline_id)

        # Perform simulations
        for i in range(num_simulations):
            print(f"performing simulation {i+1}...")
            node = root_node

            # Select
            while not node.has_untried_moves() and node.has_children():
                node = node.select_child()

            # Expand
            if node.has_untried_moves():
                move_state = node.select_untried_move()
                node = node.add_child(move_state, self._lm, self._width, self._max_depth, self._cpuct, self._newline_id)

            # Rollout
            rollout_state = list(node.state)
            while len(rollout_state) < self._max_depth and not is_sequence_complete(rollout_state, self._newline_id):
                rollout_state += [self._select_next_move_randomly(rollout_state, self._lm, self._width)]

            # Backpropagate from the expanded node and work back to the root node
            score = self._eval_function(rollout_state)
            while node is not None:
                node.visits += 1
                node.wins += score
                node = node.parent

            self._store_best(rollout_state, score)

        # return the move that was most visited
        most_visited_node = sorted(root_node.children, key=lambda c: c.visits)[-1]
        return most_visited_node.state

    def _select_next_move_randomly(self, rollout_state: List[int], language_model, width: int) -> int:
        # TODO consider regular sampling instead (i.e. torch.multinomial(probs, num_samples=1), with k=width)
        top_n_child_ids, top_n_weights = language_model.top_n_vocab_with_weights(width, rollout_state)
        return np.random.choice(top_n_child_ids, p=top_n_weights)

    def _store_best(self, rollout_state, score):
        current_best = self._best_sequence
        if current_best is None or score > current_best[1]:
            self._best_sequence = (rollout_state, score)

    def get_best_sequence(self):
        return self._best_sequence


class _LanguageModel:
    def __init__(self, model: GPT, config: GPTConfig, child_ids: List[int], device: str, temperature: float):
        self._model = model
        self._model.eval()
        self._config = config
        self._child_ids = child_ids
        self._device = device
        self._temperature = temperature

    def top_n_vocab_with_weights(self, n, token_sequence):
        idx = (torch.tensor(token_sequence, dtype=torch.long, device=self._device)[None, ...])

        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= self._config.block_size else idx[:, -self._config.block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = self._model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / self._temperature

        dist = torch.distributions.categorical.Categorical(logits=logits)

        tokens_and_log_probs = []
        for child_id in self._child_ids:
            new_token_sequence = list(token_sequence)
            new_token_sequence.append(child_id)
            log_prob = dist.log_prob(torch.tensor(child_id, device=self._device).view(1)).item()
            tokens_and_log_probs.append((child_id, log_prob))

        top_n = list(reversed(sorted(tokens_and_log_probs, key=lambda k: k[1])))[:n]

        top_n_child_ids = [t[0] for t in top_n]
        top_n_weights = self._normalize([t[1] for t in top_n])

        return top_n_child_ids, top_n_weights

    @staticmethod
    def _normalize(log_probs):
        probs = [math.exp(lp) for lp in log_probs]
        prob_factor = 1 / sum(probs)
        return [prob_factor * p for p in probs]


class _Node:
    def __init__(self, state: List[int], language_model: _LanguageModel, width, max_depth, cpuct, newline_id, parent=None):
        self.state = state
        self._cpuct = cpuct
        self._newline_id = newline_id
        self._lm = language_model
        self._width = width
        self._max_depth = max_depth
        self.wins = 0.0
        self.visits = 0.0
        self.prob = None
        self.parent = parent
        self.children = []
        self.untried_moves, self.child_weight_map = self._get_child_states()

    def _get_child_states(self):
        child_states = []
        child_state_weight_map = {}
        if len(self.state) < self._max_depth and not is_sequence_complete(self.state, self._newline_id):
            top_n_child_ids, top_n_weights = self._lm.top_n_vocab_with_weights(self._width, self.state)
            for i in range(len(top_n_child_ids)):
                child_state = self.state + [top_n_child_ids[i]]
                child_states.append(child_state)
                child_state_weight_map[tuple(child_state)] = top_n_weights[i]
        return child_states, child_state_weight_map

    def _average_value(self):
        return self.wins / self.visits

    def has_untried_moves(self):
        return self.untried_moves != []

    def select_untried_move(self):
        return random.choice(self.untried_moves)

    def add_child(self, child_state, language_model, width, max_depth, c, newline_id):
        child = _Node(child_state, language_model, width, max_depth, c, newline_id, parent=self)
        child.prob = self.child_weight_map[tuple(child_state)]
        self.children.append(child)
        self.untried_moves.remove(child_state)
        return child

    def has_children(self):
        return self.children != []

    def select_child(self):
        highest_puct = None
        selected_child_node = None
        for child_node in self.children:
            puct = child_node.puct()
            if highest_puct is None or highest_puct < puct:
                highest_puct = puct
                selected_child_node = child_node
        return selected_child_node

    def puct(self):
        if self.visits == 0:
            return math.inf
        if self.prob is None:
            raise Exception("node has no action prob: %s" % self.state)
        return self.wins / self.visits + self._cpuct * self.prob * (sqrt(self.parent.visits) / (1 + self.visits))
