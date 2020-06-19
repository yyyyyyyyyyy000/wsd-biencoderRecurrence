import torch
import torch.nn as nn
from transformers import BertModel


class glossEncoder(nn.Module):
    def __init__(self, device):
        super(glossEncoder, self).__init__()
        self.device = device
        self.model = BertModel.from_pretrained('bert-base-uncased', from_tf=True)

    def forward(self, def_li):
        # TODO: return from the gloss encoder, a list, len(list) = batch_size,   [[tensor1, tensor2, tensor3],[],[],[]]
        #  for each item in the list standing for a sentence, is a sub_list.  len(sub_list) = # polysemes in this sentence
        #  for each item in the sub_list standing for a polyseme, is a tensor. shape = # definitions for this polysemes, bert_hidden

        #  input: [[a=(), (), ().  ] [] [] [] [] ].  len(list) = 64.
        #  for each sub_list, standing for a single sentence. len(sublist) = # Polysemes in this sentence.
        #  each a in the sublist stands for a polyseme in this single sentence.
        #  there are many senses for this single sentence
        #   So a is a tuple (def_tensor,  mask_tensor).
        def_vec_li = []
        for sent_li in def_li:
            sub_list = []
            for poly_tu in sent_li:
                poly_tensors = poly_tu[0].to(self.device)
                mask_tensors = poly_tu[1].to(self.device)

                output, _ = self.model(poly_tensors, attention_mask=mask_tensors)
                # shape of output: # senses of a single polyseme (batch_size), max_sense_def, 768
                output = output[:, 0, :].cpu()  # shape: # senses of a single polyseme, 768,    a tensor
                sub_list.append(output)

            def_vec_li.append(sub_list)
        return def_vec_li
