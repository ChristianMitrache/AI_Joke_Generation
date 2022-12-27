from transformers import RobertaTokenizerFast, Seq2SeqTrainer
from torch import nn
import torch

# Defining trainer class with custom loss:
def label_smoothed_nll_loss(lprobs, target, epsilon, weights, ignore_index=None):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
        # print("got through target unsqueeze")
    nll_loss = -lprobs.gather(dim=-1, index=target)
    # print("got through gather")
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    # print("lprobs sum:")
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        # print("through pad mask")
        nll_loss.masked_fill_(pad_mask, 0.0)
        # print("nll loss masked fill")
        smooth_loss.masked_fill_(pad_mask, 0.0)
        # print("smooth loss masked fill")
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    # print("got the loss")
    # print("loss shape")
    # print(loss.shape)
    # print(loss)
    loss_batch = torch.sum(torch.squeeze(loss), dim=1)
    # print("Loss Batch: {0}".format(loss_batch))
    # print(loss_batch.shape)
    # print(weights.shape)
    # ipdb.set_trace(context = 6)
    # weighted_loss = torch.sum(loss_batch*weights)
    weighted_loss = torch.sum(loss_batch * weights) / (lprobs.shape[0] * lprobs.shape[1])
    return weighted_loss

# Should be doing forward passes before computing the conditional log probabilities, however due to lack of
# computational resources, I use the loss defined in this module to bias the generative
# distribution towards punchlines in dataset.
# Upcoming: Using a GAN style of training with a discriminator based on the hidden states of a bert model.
class CustomTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        scores = inputs.get('score').to(model.device)
        del inputs['score']
        outputs = model(**inputs)
        logits = outputs.get("logits")
        log_soft_max_func = nn.LogSoftmax(dim=2)
        lprobs = log_soft_max_func(logits)
        # Computing custom loss:
        loss = label_smoothed_nll_loss(lprobs, labels, 0.05, scores, ignore_index=1)
        return (loss, outputs) if return_outputs else loss
