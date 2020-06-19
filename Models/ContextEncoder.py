from transformers import BertModel
import torch
import torch.nn as nn




class contextEncoder(nn.Module):
    # TODO: Given a batch of data, extract the context word vectors tensors
    #  return a list, len(list) = batch_size, for each item in the list
    #  item shape = numberofpolysemesForeachelementinthebatch, bert_hidden_size
    def __init__(self, device):
        super(contextEncoder, self).__init__()
        self.device = device
        self.model = BertModel.from_pretrained('bert-base-uncased', from_tf=True).to(device)
     

    def forward(self, sents, mask_sents, pos_li):
        output, _ = self.model(sents, attention_mask=mask_sents)  # shape of output: batch, max_seq_len, 768

        context_word_list = []
        for i in range(output.size(0)):
            # TODO: deal with each item in the batch
            assert output.size(0) == len(pos_li)
            single_output = output[i]
            single_pos_li = pos_li[i]
            target_vec = None
            for mini_li in single_pos_li:
                assert output.size(2) == 768
                word_piece_repr = torch.zeros(1, output.size(2)).to(self.device)
                for t in mini_li:
                    word_piece_repr = torch.add(word_piece_repr, single_output[t])
                word_repr = word_piece_repr / len(mini_li)  # shape of word_repr:  (1, 768)
                if target_vec is None:
                    target_vec = word_repr
                else:
                    target_vec = torch.cat((target_vec, word_repr), dim=0)



            context_word_list.append(target_vec)


        return context_word_list














