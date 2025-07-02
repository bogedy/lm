from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def batch_to_labeled_samples(batch: torch.IntTensor) -> [torch.IntTensor, torch.IntTensor]:
    # implement this.
    # The batches that we get from the reader have corpus-sequences of length max-context + 1.
    # We need to translate them to input/output examples, each of which is shorter by one.
    # That is, if our input is of dimension (b x n) our output is two tensors, each of dimension (b x n-1)
    inputs = batch[:,:-1] # fix this
    labels = batch[:,1:] # fix this
    return (inputs, labels)

def compute_loss(logits, gold_labels, ignore_index=0):
    # logits size is (batch, seq_len, vocab_size)
    # gold_bales size is (batch, seq_len)
    # NOTE remember to handle padding (ignore them in loss calculation!)
    # NOTE cross-entropy expects other dimensions for logits
    # NOTE you can either use cross_entropy from PyTorch, or implement the loss on your own.


    # Transpose logits to match (batch * seq_len, vocab_size)
    logits = logits.view(-1, logits.size(-1))
    
    # Flatten gold_labels as well
    gold_labels = gold_labels.flatten()

    # Compute cross-entropy loss, ignoring padding index
    loss = F.cross_entropy(logits, gold_labels, ignore_index=ignore_index)
    return loss

