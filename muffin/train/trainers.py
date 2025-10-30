from torch import nn
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler
import os
import math
import torch
import wandb
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from transformers import Trainer
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch import Tensor
from torch.nn import Module
from utils.utils import is_main_process

from muffin.eval.muffin_inference_logp import get_batch_logps, get_batch_logps_minicpm


class ChunckedRandomSampler(Sampler[int]):
    def __init__(self, data_source, chunk_size=5000) -> None:
        self.data_source = data_source
        self.chunk_size = chunk_size

    def __iter__(self):
        n = len(self.data_source)
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        print(f'Chuncked Random Sampler seed is {seed}')
        generator = torch.Generator()
        generator.manual_seed(seed)

        for st in torch.randperm(n // self.chunk_size, generator=generator).tolist():
            base = st * self.chunk_size
            for i in torch.randperm(self.chunk_size, generator=generator).tolist():
                yield base + i

        base = (n // self.chunk_size) * self.chunk_size
        for i in torch.randperm(n % self.chunk_size, generator=generator).tolist():
            yield base + i

    def __len__(self) -> int:
        return len(self.data_source)

class ZephyrTrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None:
            return None

        # Build the sampler.
        return RandomSampler(self.train_dataset)
        # return SequentialSampler(self.train_dataset)

        # if self.args.group_by_length:
        #     assert NotImplementedError
        # else:
        #     if len(self.train_dataset) >= 50_000_000:
        #         return ChunckedRandomSampler(self.train_dataset)
        #     else:
        #         # print(f'Data set size is :{len(self.train_dataset)}', flush=True)
        #         # return SequentialSampler(self.train_dataset)

        #         print(f'Shuffle Data set size is :{len(self.train_dataset)}', flush=True)
        #         return RandomSampler(self.train_dataset)

def forward_DPO(model, input_ids, labels, attention_mask, images, **kwargs):
    token_weighted = kwargs.pop('token_weighted', False)
    dpo_use_average = kwargs.pop('dpo_use_average', False)
    is_minicpm = kwargs.pop('is_minicpm', False)

    output = model(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        images=images,
        **kwargs
    )
    impl = get_batch_logps_minicpm if is_minicpm else get_batch_logps
    if token_weighted:
        token_log_prob = impl(
            output.logits, labels, return_per_token_logp=True)
        return token_log_prob
    else:
        log_prob, average_log_prob = impl(
            output.logits, labels, return_per_token_logp=False)
        if dpo_use_average:
            return average_log_prob
        return log_prob


def dpo_loss(policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             beta: float,
             reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps -
                             reference_chosen_logps).detach()
    rejected_rewards = beta * \
        (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards

def compute_weighted_logp(per_token_logp, labels, token_weight, use_average):
    loss_mask = (labels[:, 1:].clone() != -100)
    # print(f'compute wlogp {labels.shape} {loss_mask.shape}, {token_weight.shape}, {per_token_logp.shape}')
    weighted_mask = token_weight * loss_mask
    logp = (per_token_logp * weighted_mask).sum(-1)

    average_logp = logp / weighted_mask.sum(-1)
    if use_average:
        return average_logp
    return logp


def collect_preference_metrics(metrics, task,
                               chosen_rewards, rejected_rewards,
                               policy_rej_logp, policy_win_logp,
                               ref_rej_logp, ref_win_logp, reward_accuracies,
                               preprocess_func,
                               ):
    t = task
    metrics = {}
    metrics[f'rewards_{t}/chosen'] = preprocess_func(chosen_rewards)
    metrics[f'rewards_{t}/rejected'] = preprocess_func(rejected_rewards)
    metrics[f'logps_{t}/rejected'] = preprocess_func(policy_rej_logp)
    metrics[f'logps_{t}/chosen'] = preprocess_func(policy_win_logp)
    metrics[f'logps_{t}/ref_rejected'] = preprocess_func(ref_rej_logp)
    metrics[f'logps_{t}/ref_chosen'] = preprocess_func(ref_win_logp)
    metrics[f'rewards_{t}/accuracies'] = preprocess_func(
        reward_accuracies)
    metrics[f'rewards_{t}/margins'] = metrics[f'rewards_{t}/chosen'] - \
        metrics[f'rewards_{t}/rejected']
    return metrics


def get_beta_and_logps(data_dict, model, args, is_minicpm=False, is_llava15=False):
    win_input_ids = data_dict.pop('win_input_ids')
    rej_input_ids = data_dict.pop('rej_input_ids')

    win_labels = data_dict.pop('win_labels')
    rej_labels = data_dict.pop('rej_labels')

    win_attention_mask = data_dict.pop('win_attention_mask')
    rej_attention_mask = data_dict.pop('rej_attention_mask')

    ref_win_avg_logp = data_dict.pop('ref_win_avg_logp')
    ref_rej_avg_logp = data_dict.pop('ref_rej_avg_logp')
    ref_win_logp = data_dict.pop('ref_win_logp')
    ref_rej_logp = data_dict.pop('ref_rej_logp')
    ref_win_per_token_logp = data_dict.pop('ref_win_per_token_logp')
    ref_rej_per_token_logp = data_dict.pop('ref_rej_per_token_logp')
    if args.dpo_use_average:
        ref_win_logp = ref_win_avg_logp
        ref_rej_logp = ref_rej_avg_logp

    beta = data_dict.pop('beta')
    if args.task == 'DPO':
        images = data_dict.pop('images')
        if is_minicpm:
            # print(data_dict.keys())
            data_dict.pop('win_context_ids')
            data_dict.pop('rej_context_ids')
            concatenated_images = images
        else:
            concatenated_images = torch.cat([images, images], dim=0)
    elif args.task == 'KTO':
        win_images = data_dict.pop('win_images')
        rej_images = data_dict.pop('rej_images')
        concatenated_images = torch.cat([win_images, rej_images], dim=0)

    concatenated_input_ids = data_dict.pop('concatenated_input_ids')
    concatenated_labels = data_dict.pop('concatenated_labels')
    concatenated_attention_mask = data_dict.pop('concatenated_attention_mask')
    concatenated_attention_mask = None

    win_token_weight = data_dict.pop('win_token_weight')
    rej_token_weight = data_dict.pop('rej_token_weight')
    concatenated_token_weight = data_dict.pop('concatenated_token_weight')

    if is_llava15:
        (
            _,
            _,
            _,
            _,
            concatenated_inputs_embeds,
            concatenated_labels
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids=concatenated_input_ids,
            position_ids=None,
            attention_mask=None,
            past_key_values=None,
            labels=concatenated_labels,
            images=concatenated_images,
        )
        output = model.forward(
            inputs_embeds=concatenated_inputs_embeds,
            labels=None,
            **data_dict,
        )
        log_prob, average_log_prob = get_batch_logps(
            output.logits, concatenated_labels, return_per_token_logp=False)
        if args.dpo_use_average:
            concatenated_logp = average_log_prob
        else:
            concatenated_logp =log_prob
    else:
        concatenated_logp = forward_DPO(model,
                                        concatenated_input_ids,
                                        concatenated_labels,
                                        concatenated_attention_mask,
                                        concatenated_images,
                                        token_weighted=args.dpo_token_weighted,
                                        dpo_use_average=args.dpo_use_average,
                                        is_minicpm=is_minicpm,
                                        **data_dict)
    win_size = win_input_ids.shape[0]
    rej_size = rej_input_ids.shape[0]
    assert win_size == rej_size

    if args.dpo_token_weighted:
        if is_llava15:
            raise NotImplementedError
        # print(f'compute_loss win {win_input_ids.shape} {win_labels.shape} {ref_win_per_token_logp.shape} {win_token_weight.shape}', flush=True)
        # print(f'compute_loss rej {rej_input_ids.shape} {rej_labels.shape} {ref_rej_per_token_logp.shape} {rej_token_weight.shape}', flush=True)
        # print(f'compute_loss cat {concatenated_input_ids.shape} {concatenated_labels.shape} {concatenated_logp.shape} {concatenated_token_weight.shape}', flush=True)

        # for i in range(len(ref_win_per_token_logp)):
        #     print(f'compuate loss {i} win_input_ids={win_input_ids[i]}\nwin_labels={win_labels[i]}\nwin_per_token_logp={ref_win_per_token_logp[i]}\nwin_token_weight={win_token_weight[i]}\n', flush=True)
        #     print(f'compuate loss {i} rej_input_ids={rej_input_ids[i]}\nrej_labels={rej_labels[i]}\nrej_per_token_logp={ref_rej_per_token_logp[i]}\nrej_token_weight={rej_token_weight[i]}\n', flush=True)
        ref_win_logp = compute_weighted_logp(
            ref_win_per_token_logp, win_labels, win_token_weight, args.dpo_use_average)
        ref_rej_logp = compute_weighted_logp(
            ref_rej_per_token_logp, rej_labels, rej_token_weight, args.dpo_use_average)
        concatenated_logp = compute_weighted_logp(
            concatenated_logp, concatenated_labels, concatenated_token_weight, args.dpo_use_average)

        if torch.any(torch.isnan(ref_win_logp)):
            print(f'ref_win_logp fail', flush=True)
            exit()
        if torch.any(torch.isnan(ref_rej_logp)):
            print(f'ref_rej_logp fail', flush=True)
            exit()
        if torch.any(torch.isnan(concatenated_logp)):
            print(f'concatenated_logp fail', flush=True)
            exit()

    policy_win_logp, policy_rej_logp = concatenated_logp.split(
        [win_size, rej_size])
    return policy_win_logp, policy_rej_logp, ref_win_logp, ref_rej_logp, beta



class LLaVA15DPOTrainer(ZephyrTrainer):

    def compute_loss(self, model: Module, inputs: dict, return_outputs=False):
        if self.args.past_index >= 0:
            raise NotImplementedError

        def gather_and_do_mean(x):
            return self._nested_gather(x.mean()).mean().item()

        data_dict = inputs
        policy_win_logp, policy_rej_logp, ref_win_logp, ref_rej_logp, beta = get_beta_and_logps(
            data_dict, model, self.args, is_llava15=True)

        losses, chosen_rewards, rejected_rewards = dpo_loss(policy_win_logp,
                                                            policy_rej_logp,
                                                            ref_win_logp,
                                                            ref_rej_logp,
                                                            beta=beta)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        SFT_weight = float(os.environ.get('SFT_weight', 0.0))
        DPO_weight = float(os.environ.get('DPO_weight', 1.0))
        loss = DPO_weight * losses.mean() - SFT_weight * policy_win_logp.mean()

        t = 'train' if model.training else 'test'
        metrics = {}
        metrics = collect_preference_metrics(metrics, t, chosen_rewards, rejected_rewards,
                                             policy_rej_logp, policy_win_logp,
                                             ref_rej_logp, ref_win_logp, reward_accuracies,
                                             gather_and_do_mean)
        self.log(metrics)

        return loss


def kto_loss(policy_desirable_logps: torch.FloatTensor,
             policy_undesirable_logps: torch.FloatTensor,
             reference_desirable_logps: torch.FloatTensor,
             reference_undesirable_logps: torch.FloatTensor,
             beta: float,
             desirability_weight: float = 1.0,
             undesirability_weight: float = 1.0) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the KTO loss for a batch of policy and reference model log probabilities.
    
    KTO uses Kahneman-Tversky optimization based on prospect theory.
    
    Args:
        policy_desirable_logps: Log probabilities of the policy model for desirable responses. Shape: (batch_size,)
        policy_undesirable_logps: Log probabilities of the policy model for undesirable responses. Shape: (batch_size,)
        reference_desirable_logps: Log probabilities of the reference model for desirable responses. Shape: (batch_size,)
        reference_undesirable_logps: Log probabilities of the reference model for undesirable responses. Shape: (batch_size,)
        beta: Temperature parameter for the KTO loss
        desirability_weight: Weight for desirable examples (lambda_D)
        undesirability_weight: Weight for undesirable examples (lambda_U)
        
    Returns:
        A tuple of four tensors: (losses, desirable_rewards, undesirable_rewards, kl_penalties)
    """
    # KL divergence between policy and reference for both types
    kl_desirable = policy_desirable_logps - reference_desirable_logps
    kl_undesirable = policy_undesirable_logps - reference_undesirable_logps
    
    # KTO uses a utility-based formulation
    # For desirable outputs: we want to maximize utility (minimize loss)
    # For undesirable outputs: we want to minimize utility (maximize loss aversion)
    
    # Desirable loss: we want KL to be positive (policy better than reference)
    desirable_losses = desirability_weight * (1 - F.sigmoid(beta * kl_desirable))
    
    # Undesirable loss: we want KL to be negative (policy worse than reference)  
    undesirable_losses = undesirability_weight * (1 - F.sigmoid(-beta * kl_undesirable))
    
    # Combine losses
    losses = torch.cat([desirable_losses, undesirable_losses], dim=0)
    
    # Compute implicit rewards
    desirable_rewards = beta * kl_desirable.detach()
    undesirable_rewards = beta * kl_undesirable.detach()
    
    return losses, desirable_losses, undesirable_losses, desirable_rewards, undesirable_rewards


def simpo_loss(policy_chosen_logps: torch.FloatTensor,
               policy_rejected_logps: torch.FloatTensor,
               beta: float,
               gamma: float = 0.5,
               length_normalize: bool = True,
               chosen_lengths: Optional[torch.FloatTensor] = None,
               rejected_lengths: Optional[torch.FloatTensor] = None) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the SimPO loss for a batch of policy log probabilities.
    
    SimPO is a reference-free variant of DPO that uses average log probability as implicit reward.
    
    Args:
        policy_chosen_logps: Log probabilities of the policy model for chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the SimPO loss
        gamma: Target reward margin (default: 0.5)
        length_normalize: Whether to use length normalization (default: True)
        chosen_lengths: Sequence lengths for chosen responses (required if length_normalize=True)
        rejected_lengths: Sequence lengths for rejected responses (required if length_normalize=True)
        
    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards)
    """
    
    # If length normalization is enabled, normalize log probs by sequence length
    if length_normalize:
        if chosen_lengths is None or rejected_lengths is None:
            raise ValueError("chosen_lengths and rejected_lengths must be provided when length_normalize=True")
        
        # Average log probability (implicit reward)
        policy_chosen_rewards = policy_chosen_logps / chosen_lengths
        policy_rejected_rewards = policy_rejected_logps / rejected_lengths
    else:
        policy_chosen_rewards = policy_chosen_logps
        policy_rejected_rewards = policy_rejected_logps
    
    # SimPO loss with target reward margin
    # Loss = -log(sigmoid(beta * (r_chosen - r_rejected - gamma)))
    logits = beta * (policy_chosen_rewards - policy_rejected_rewards - gamma)
    losses = -F.logsigmoid(logits)
    
    # Return rewards for logging
    chosen_rewards = policy_chosen_rewards.detach()
    rejected_rewards = policy_rejected_rewards.detach()
    
    return losses, chosen_rewards, rejected_rewards


def compute_sequence_length(labels: torch.Tensor) -> torch.FloatTensor:
    """Compute the actual sequence length (excluding padding tokens marked as -100)."""
    return (labels != -100).sum(dim=-1).float()


class LLaVA15KTOTrainer(ZephyrTrainer):
    def compute_loss(self, model: Module, inputs: dict, return_outputs=False):
        if self.args.past_index >= 0:
            raise NotImplementedError
        
        def gather_and_do_mean(x):
            return self._nested_gather(x.mean()).mean().item()
        
        data_dict = inputs
        
        # Extract desirable and undesirable data
        desirable_input_ids = data_dict.pop('win_input_ids')
        undesirable_input_ids = data_dict.pop('rej_input_ids')
        desirable_labels = data_dict.pop('win_labels')
        undesirable_labels = data_dict.pop('rej_labels')
        desirable_attention_mask = data_dict.pop('win_attention_mask')
        undesirable_attention_mask = data_dict.pop('rej_attention_mask')
        
        # Reference model log probabilities
        ref_desirable_avg_logp = data_dict.pop('ref_win_avg_logp')
        ref_undesirable_avg_logp = data_dict.pop('ref_rej_avg_logp')
        ref_desirable_logp = data_dict.pop('ref_win_logp')
        ref_undesirable_logp = data_dict.pop('ref_rej_logp')
        
        if self.args.dpo_use_average:
            ref_desirable_logp = ref_desirable_avg_logp
            ref_undesirable_logp = ref_undesirable_avg_logp
        
        beta = data_dict.pop('beta')
        images = data_dict.pop('images')
        
        # Concatenate inputs for efficient forward pass
        concatenated_images = torch.cat([images, images], dim=0)
        concatenated_input_ids = data_dict.pop('concatenated_input_ids')
        concatenated_labels = data_dict.pop('concatenated_labels')
        
        # Forward pass through model
        (
            _,
            _,
            _,
            _,
            concatenated_inputs_embeds,
            concatenated_labels
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids=concatenated_input_ids,
            position_ids=None,
            attention_mask=None,
            past_key_values=None,
            labels=concatenated_labels,
            images=concatenated_images,
        )
        
        output = model.forward(
            inputs_embeds=concatenated_inputs_embeds,
            labels=None,
            **data_dict,
        )
        
        log_prob, average_log_prob = get_batch_logps(
            output.logits, concatenated_labels, return_per_token_logp=False)
        
        if self.args.dpo_use_average:
            concatenated_logp = average_log_prob
        else:
            concatenated_logp = log_prob
        
        desirable_size = desirable_input_ids.shape[0]
        undesirable_size = undesirable_input_ids.shape[0]
        
        policy_desirable_logp, policy_undesirable_logp = concatenated_logp.split(
            [desirable_size, undesirable_size])
        
        # Compute KTO loss
        losses, desirable_losses, undesirable_losses, desirable_rewards, undesirable_rewards = kto_loss(
            policy_desirable_logp,
            policy_undesirable_logp,
            ref_desirable_logp,
            ref_undesirable_logp,
            beta=beta,
            desirability_weight=getattr(self.args, 'kto_desirable_weight', 1.0),
            undesirability_weight=getattr(self.args, 'kto_undesirable_weight', 1.0)
        )
        
        loss = losses.mean()
        
        # Logging metrics
        t = 'train' if model.training else 'test'
        metrics = {}
        metrics[f'rewards_{t}/desirable'] = gather_and_do_mean(desirable_rewards)
        metrics[f'rewards_{t}/undesirable'] = gather_and_do_mean(undesirable_rewards)
        metrics[f'logps_{t}/desirable'] = gather_and_do_mean(policy_desirable_logp)
        metrics[f'logps_{t}/undesirable'] = gather_and_do_mean(policy_undesirable_logp)
        metrics[f'logps_{t}/ref_desirable'] = gather_and_do_mean(ref_desirable_logp)
        metrics[f'logps_{t}/ref_undesirable'] = gather_and_do_mean(ref_undesirable_logp)
        metrics[f'loss_{t}/desirable'] = gather_and_do_mean(desirable_losses)
        metrics[f'loss_{t}/undesirable'] = gather_and_do_mean(undesirable_losses)
        
        self.log(metrics)
        
        return loss


class LLaVA15SimPOTrainer(ZephyrTrainer):
    def compute_loss(self, model: Module, inputs: dict, return_outputs=False):
        if self.args.past_index >= 0:
            raise NotImplementedError
        
        def gather_and_do_mean(x):
            return self._nested_gather(x.mean()).mean().item()
        
        data_dict = inputs
        
        win_input_ids = data_dict.pop('win_input_ids')
        rej_input_ids = data_dict.pop('rej_input_ids')
        win_labels = data_dict.pop('win_labels')
        rej_labels = data_dict.pop('rej_labels')
        win_attention_mask = data_dict.pop('win_attention_mask')
        rej_attention_mask = data_dict.pop('rej_attention_mask')
        
        # SimPO doesn't use reference model, so we can ignore these
        data_dict.pop('ref_win_avg_logp', None)
        data_dict.pop('ref_rej_avg_logp', None)
        data_dict.pop('ref_win_logp', None)
        data_dict.pop('ref_rej_logp', None)
        data_dict.pop('ref_win_per_token_logp', None)
        data_dict.pop('ref_rej_per_token_logp', None)
        
        beta = data_dict.pop('beta')
        images = data_dict.pop('images')
        
        concatenated_images = torch.cat([images, images], dim=0)
        concatenated_input_ids = data_dict.pop('concatenated_input_ids')
        concatenated_labels = data_dict.pop('concatenated_labels')
        
        data_dict.pop('win_token_weight', None)
        data_dict.pop('rej_token_weight', None)
        data_dict.pop('concatenated_token_weight', None)
        
        # Forward pass
        (
            _,
            _,
            _,
            _,
            concatenated_inputs_embeds,
            concatenated_labels
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids=concatenated_input_ids,
            position_ids=None,
            attention_mask=None,
            past_key_values=None,
            labels=concatenated_labels,
            images=concatenated_images,
        )
        
        output = model.forward(
            inputs_embeds=concatenated_inputs_embeds,
            labels=None,
            **data_dict,
        )
        
        # Get log probabilities (NOT normalized by length yet)
        log_prob, _ = get_batch_logps(
            output.logits, concatenated_labels, return_per_token_logp=False)
        
        win_size = win_input_ids.shape[0]
        rej_size = rej_input_ids.shape[0]
        
        policy_win_logp, policy_rej_logp = log_prob.split([win_size, rej_size])
        
        # Compute sequence lengths for length normalization
        win_lengths = compute_sequence_length(win_labels)
        rej_lengths = compute_sequence_length(rej_labels)
        
        # SimPO loss with length normalization and target margin
        gamma = getattr(self.args, 'simpo_gamma', 0.5)
        losses, chosen_rewards, rejected_rewards = simpo_loss(
            policy_win_logp,
            policy_rej_logp,
            beta=beta,
            gamma=gamma,
            length_normalize=True,
            chosen_lengths=win_lengths,
            rejected_lengths=rej_lengths
        )
        
        loss = losses.mean()
        
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        # Logging
        t = 'train' if model.training else 'test'
        metrics = {}
        metrics[f'rewards_{t}/chosen'] = gather_and_do_mean(chosen_rewards)
        metrics[f'rewards_{t}/rejected'] = gather_and_do_mean(rejected_rewards)
        metrics[f'logps_{t}/chosen'] = gather_and_do_mean(policy_win_logp)
        metrics[f'logps_{t}/rejected'] = gather_and_do_mean(policy_rej_logp)
        metrics[f'rewards_{t}/accuracies'] = gather_and_do_mean(reward_accuracies)
        metrics[f'rewards_{t}/margins'] = metrics[f'rewards_{t}/chosen'] - metrics[f'rewards_{t}/rejected']
        metrics[f'lengths/chosen'] = gather_and_do_mean(win_lengths)
        metrics[f'lengths/rejected'] = gather_and_do_mean(rej_lengths)
        
        self.log(metrics)
        
        return loss


def orpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    beta: float,
    alpha: float = 1.0
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Compute the ORPO loss for a batch of policy log probabilities.
    
    ORPO combines SFT loss with odds ratio-based preference optimization.
    No reference model needed!
    
    Args:
        policy_chosen_logps: Log probabilities of chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of rejected responses. Shape: (batch_size,)
        beta: Weight for the odds ratio loss term
        alpha: Weight for the SFT loss term (default: 1.0)
        
    Returns:
        Tuple of (total_losses, sft_losses, or_losses, log_odds_ratio)
    """
    # SFT loss: Negative log likelihood of chosen responses
    sft_losses = -policy_chosen_logps
    
    # Compute log odds for chosen and rejected
    # log_odds = log(p / (1-p)) = log(p) - log(1-p)
    # For numerical stability, use log_odds ≈ logp when using average logp
    log_odds_chosen = policy_chosen_logps - torch.log1p(-torch.exp(policy_chosen_logps).clamp(max=0.9999))
    log_odds_rejected = policy_rejected_logps - torch.log1p(-torch.exp(policy_rejected_logps).clamp(max=0.9999))
    
    # Log odds ratio
    log_odds_ratio = log_odds_chosen - log_odds_rejected
    
    # ORPO loss: OR loss encourages log odds ratio > 0
    or_losses = -F.logsigmoid(log_odds_ratio)
    
    # Combined loss
    total_losses = alpha * sft_losses + beta * or_losses
    
    return total_losses, sft_losses, or_losses, log_odds_ratio.detach()


def sppo_loss_iteration(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    preference_model_scores: torch.FloatTensor,
    beta: float,
    gamma: float = 0.5,
    length_normalize: bool = True,
    chosen_lengths: Optional[torch.FloatTensor] = None,
    rejected_lengths: Optional[torch.FloatTensor] = None
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Compute one iteration of SPPO loss using self-play with a preference model.
    
    SPPO uses Nash equilibrium from game theory with iterative self-play.
    
    Args:
        policy_chosen_logps: Log probabilities of chosen responses
        policy_rejected_logps: Log probabilities of rejected responses  
        preference_model_scores: Scores from preference oracle (positive = chosen better)
        beta: Temperature parameter
        gamma: Target reward margin
        length_normalize: Whether to use length normalization
        chosen_lengths: Sequence lengths for chosen (required if length_normalize=True)
        rejected_lengths: Sequence lengths for rejected (required if length_normalize=True)
        
    Returns:
        Tuple of (losses, chosen_rewards, rejected_rewards)
    """
    # Length normalization if enabled
    if length_normalize:
        if chosen_lengths is None or rejected_lengths is None:
            raise ValueError("chosen_lengths and rejected_lengths required when length_normalize=True")
        policy_chosen_rewards = policy_chosen_logps / chosen_lengths
        policy_rejected_rewards = policy_rejected_logps / rejected_lengths
    else:
        policy_chosen_rewards = policy_chosen_logps
        policy_rejected_rewards = policy_rejected_logps
    
    # SPPO uses preference model to reweight examples
    # Higher preference score means we trust this preference more
    preference_weights = torch.sigmoid(preference_model_scores)
    
    # Compute logits with margin and apply preference weighting
    logits = beta * (policy_chosen_rewards - policy_rejected_rewards - gamma)
    
    # Weighted loss based on preference model confidence
    losses = -preference_weights * F.logsigmoid(logits)
    
    chosen_rewards = policy_chosen_rewards.detach()
    rejected_rewards = policy_rejected_rewards.detach()
    
    return losses, chosen_rewards, rejected_rewards

class LLaVA15ORPOTrainer(ZephyrTrainer):
    """
    ORPO Trainer - Monolithic preference optimization without reference model.
    Combines SFT + alignment in a single phase.
    """
    def compute_loss(self, model: Module, inputs: dict, return_outputs=False):
        if self.args.past_index >= 0:
            raise NotImplementedError
        
        def gather_and_do_mean(x):
            return self._nested_gather(x.mean()).mean().item()
        
        data_dict = inputs
        
        win_input_ids = data_dict.pop('win_input_ids')
        rej_input_ids = data_dict.pop('rej_input_ids')
        win_labels = data_dict.pop('win_labels')
        rej_labels = data_dict.pop('rej_labels')
        win_attention_mask = data_dict.pop('win_attention_mask')
        rej_attention_mask = data_dict.pop('rej_attention_mask')
        
        # ORPO doesn't use reference model
        data_dict.pop('ref_win_avg_logp', None)
        data_dict.pop('ref_rej_avg_logp', None)
        data_dict.pop('ref_win_logp', None)
        data_dict.pop('ref_rej_logp', None)
        data_dict.pop('ref_win_per_token_logp', None)
        data_dict.pop('ref_rej_per_token_logp', None)
        
        beta = data_dict.pop('beta')
        images = data_dict.pop('images')
        
        concatenated_images = torch.cat([images, images], dim=0)
        concatenated_input_ids = data_dict.pop('concatenated_input_ids')
        concatenated_labels = data_dict.pop('concatenated_labels')
        
        data_dict.pop('win_token_weight', None)
        data_dict.pop('rej_token_weight', None)
        data_dict.pop('concatenated_token_weight', None)
        
        # Forward pass
        (
            _,
            _,
            _,
            _,
            concatenated_inputs_embeds,
            concatenated_labels
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids=concatenated_input_ids,
            position_ids=None,
            attention_mask=None,
            past_key_values=None,
            labels=concatenated_labels,
            images=concatenated_images,
        )
        
        output = model.forward(
            inputs_embeds=concatenated_inputs_embeds,
            labels=None,
            **data_dict,
        )
        
        # Get average log probabilities (important for ORPO)
        log_prob, average_log_prob = get_batch_logps(
            output.logits, concatenated_labels, return_per_token_logp=False)
        
        # Use average log prob for odds calculation
        concatenated_logp = average_log_prob
        
        win_size = win_input_ids.shape[0]
        rej_size = rej_input_ids.shape[0]
        
        policy_win_logp, policy_rej_logp = concatenated_logp.split([win_size, rej_size])
        
        # ORPO loss
        alpha = getattr(self.args, 'orpo_alpha', 1.0)  # SFT weight
        total_losses, sft_losses, or_losses, log_odds_ratio = orpo_loss(
            policy_win_logp,
            policy_rej_logp,
            beta=beta,
            alpha=alpha
        )
        
        loss = total_losses.mean()
        
        # Compute implicit rewards from log odds ratio
        chosen_rewards = log_odds_ratio
        rejected_rewards = -log_odds_ratio
        reward_accuracies = (log_odds_ratio > 0).float()
        
        # Logging
        t = 'train' if model.training else 'test'
        metrics = {}
        metrics[f'loss_{t}/total'] = gather_and_do_mean(total_losses)
        metrics[f'loss_{t}/sft'] = gather_and_do_mean(sft_losses)
        metrics[f'loss_{t}/or'] = gather_and_do_mean(or_losses)
        metrics[f'rewards_{t}/chosen'] = gather_and_do_mean(chosen_rewards)
        metrics[f'rewards_{t}/rejected'] = gather_and_do_mean(rejected_rewards)
        metrics[f'rewards_{t}/accuracies'] = gather_and_do_mean(reward_accuracies)
        metrics[f'rewards_{t}/margins'] = metrics[f'rewards_{t}/chosen'] - metrics[f'rewards_{t}/rejected']
        metrics[f'logps_{t}/chosen'] = gather_and_do_mean(policy_win_logp)
        metrics[f'logps_{t}/rejected'] = gather_and_do_mean(policy_rej_logp)
        metrics[f'log_odds_ratio_{t}'] = gather_and_do_mean(log_odds_ratio)
        
        self.log(metrics)
        
        return loss

class LLaVA15SPPOTrainer(ZephyrTrainer):
    """
    SPPO Trainer - Self-Play Preference Optimization using game theory.
    Requires a small preference model (oracle) for scoring pairs.
    """
    def __init__(self, *args, preference_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.preference_model = preference_model
        if self.preference_model is not None:
            self.preference_model.eval()
            for param in self.preference_model.parameters():
                param.requires_grad = False
    
    def compute_preference_scores(self, images, win_inputs, rej_inputs, win_labels, rej_labels):
        """
        Compute preference scores using a small preference oracle model.
        Returns positive scores when chosen is preferred.
        """
        if self.preference_model is None:
            # Fallback: use simple heuristic based on response length
            win_lengths = (win_labels != -100).sum(dim=-1).float()
            rej_lengths = (rej_labels != -100).sum(dim=-1).float()
            # Prefer responses that are not too short or too long
            optimal_length = 50.0
            win_penalty = torch.abs(win_lengths - optimal_length) / optimal_length
            rej_penalty = torch.abs(rej_lengths - optimal_length) / optimal_length
            scores = rej_penalty - win_penalty  # Positive when chosen is better
            return scores
        
        # Use actual preference model
        with torch.no_grad():
            # Forward pass for chosen
            win_outputs = self.preference_model(
                input_ids=win_inputs,
                attention_mask=(win_labels != -100),
                images=images
            )
            
            # Forward pass for rejected  
            rej_outputs = self.preference_model(
                input_ids=rej_inputs,
                attention_mask=(rej_labels != -100),
                images=images
            )
            
            # Extract preference scores (assuming model outputs logits)
            # Positive score = chosen preferred
            win_scores = win_outputs.logits[:, -1, :].mean(dim=-1)
            rej_scores = rej_outputs.logits[:, -1, :].mean(dim=-1)
            preference_scores = win_scores - rej_scores
            
        return preference_scores
    
    def compute_loss(self, model: Module, inputs: dict, return_outputs=False):
        if self.args.past_index >= 0:
            raise NotImplementedError
        
        def gather_and_do_mean(x):
            return self._nested_gather(x.mean()).mean().item()
        
        data_dict = inputs
        
        win_input_ids = data_dict.pop('win_input_ids')
        rej_input_ids = data_dict.pop('rej_input_ids')
        win_labels = data_dict.pop('win_labels')
        rej_labels = data_dict.pop('rej_labels')
        win_attention_mask = data_dict.pop('win_attention_mask')
        rej_attention_mask = data_dict.pop('rej_attention_mask')
        
        # SPPO doesn't use reference model from data
        data_dict.pop('ref_win_avg_logp', None)
        data_dict.pop('ref_rej_avg_logp', None)
        data_dict.pop('ref_win_logp', None)
        data_dict.pop('ref_rej_logp', None)
        data_dict.pop('ref_win_per_token_logp', None)
        data_dict.pop('ref_rej_per_token_logp', None)
        
        beta = data_dict.pop('beta')
        images = data_dict.pop('images')
        
        # Compute preference scores from oracle
        preference_scores = self.compute_preference_scores(
            images, win_input_ids, rej_input_ids, win_labels, rej_labels
        )
        
        concatenated_images = torch.cat([images, images], dim=0)
        concatenated_input_ids = data_dict.pop('concatenated_input_ids')
        concatenated_labels = data_dict.pop('concatenated_labels')
        
        data_dict.pop('win_token_weight', None)
        data_dict.pop('rej_token_weight', None)
        data_dict.pop('concatenated_token_weight', None)
        
        # Forward pass
        (
            _,
            _,
            _,
            _,
            concatenated_inputs_embeds,
            concatenated_labels
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids=concatenated_input_ids,
            position_ids=None,
            attention_mask=None,
            past_key_values=None,
            labels=concatenated_labels,
            images=concatenated_images,
        )
        
        output = model.forward(
            inputs_embeds=concatenated_inputs_embeds,
            labels=None,
            **data_dict,
        )
        
        # Get log probabilities
        log_prob, _ = get_batch_logps(
            output.logits, concatenated_labels, return_per_token_logp=False)
        
        win_size = win_input_ids.shape[0]
        rej_size = rej_input_ids.shape[0]
        
        policy_win_logp, policy_rej_logp = log_prob.split([win_size, rej_size])
        
        # Compute sequence lengths for length normalization
        win_lengths = compute_sequence_length(win_labels)
        rej_lengths = compute_sequence_length(rej_labels)
        
        # SPPO loss with preference model weighting
        gamma = getattr(self.args, 'sppo_gamma', 0.5)
        losses, chosen_rewards, rejected_rewards = sppo_loss_iteration(
            policy_win_logp,
            policy_rej_logp,
            preference_scores,
            beta=beta,
            gamma=gamma,
            length_normalize=True,
            chosen_lengths=win_lengths,
            rejected_lengths=rej_lengths
        )
        
        loss = losses.mean()
        
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        # Logging
        t = 'train' if model.training else 'test'
        metrics = {}
        metrics[f'rewards_{t}/chosen'] = gather_and_do_mean(chosen_rewards)
        metrics[f'rewards_{t}/rejected'] = gather_and_do_mean(rejected_rewards)
        metrics[f'logps_{t}/chosen'] = gather_and_do_mean(policy_win_logp)
        metrics[f'logps_{t}/rejected'] = gather_and_do_mean(policy_rej_logp)
        metrics[f'rewards_{t}/accuracies'] = gather_and_do_mean(reward_accuracies)
        metrics[f'rewards_{t}/margins'] = metrics[f'rewards_{t}/chosen'] - metrics[f'rewards_{t}/rejected']
        metrics[f'preference_scores_{t}'] = gather_and_do_mean(preference_scores)
        metrics[f'lengths/chosen'] = gather_and_do_mean(win_lengths)
        metrics[f'lengths/rejected'] = gather_and_do_mean(rej_lengths)
        
        self.log(metrics)
        
        return loss
