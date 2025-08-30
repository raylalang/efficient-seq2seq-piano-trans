import torch


def get_sequence_lengths ( sequences, eos_token_id):
    """ Get the output sequences lengths by finding the first EOS token in each sequence.
    This function assumes that the sequences are end with a specific EOS token ID.
    If the EOS token is not found in a sequence, the length of the sequence is returned.
    Args:
        sequences (torch.tensor): [Batch, Seq_len]
        eos_token_id (int): EOS token id
    Returns:
        eos_pos (list): [Batch,] EOS token id or length of the sequence if no eos in the sequence.
    """
    
    assert len(sequences.size()) == 2, "sequences should be 2D tensor."
    assert eos_token_id >= 0, "eos_token_id should be a positive integer."
    # assert eos_token_id < sequences.size(1), "eos_token_id should be less than the length of the sequence."
    # Mask: where is EOS
    eos_mask = (sequences == eos_token_id)
    
    B, T = sequences.size()

    # Use `max` on reversed cumsum to find first EOS index
    # Replace False with a large index to avoid it being selected
    indices = torch.arange(sequences.size(1)).expand_as(sequences).flatten()
    
    indices[~eos_mask.flatten()] = sequences.size(1) # Should be flattened to a 1D tensor.
    eos_pos = indices.reshape([B, T]).min(dim=1).values

    return eos_pos