from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, BertModel
import torch
from torch.nn.utils.rnn import pad_sequence
from Models.GlossEncoder import glossEncoder
from tqdm import tqdm
# device = torch.device('cuda: 1')
#
# gloss_encoder = BertModel.from_pretrained('bert-base-uncased', from_tf=True).to(device)

SynsetsDict = {}
SynsetsDefinitions = {}

all_synsets = list(wn.all_synsets())

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# for synset in tqdm(all_synsets):
#     tokenized_definitions = tokenizer.tokenize(synset.definition())
#     tokenized_definitions = ['[cls]'] + tokenized_definitions
#     input_definitions = tokenizer.convert_tokens_to_ids(tokenized_definitions)  # type(input_definitions) = list
#     SynsetsDefinitions[synset.name()] = torch.tensor(input_definitions)
# torch.save(SynsetsDefinitions, 'definitions.data')
SynsetsDefinitions = torch.load('definitions.data')
count = 0
for k, (i, j) in tqdm(enumerate(SynsetsDefinitions.items())):

    if j.size(0) > 64:
        print(count,k,j.size(0))
        count += 1


# def updateSynsetsEmbeddings():
#     # the structure of the SynsetsDict is synset name: Definition Embedding vectors
#     total_synsets = len(all_synsets)
#     batch_size = 8
#     batch_number = total_synsets // batch_size
#     res_batch_idx = batch_number * batch_size
#     for j,i in tqdm(enumerate(range(batch_number))):
#         print(j)
#         temp_synset_list = all_synsets[i * batch_size:(i + 1) * batch_size]
#         tensor_list = [SynsetsDefinitions[syn.name()] for syn in temp_synset_list]
#
#         processed_def = pad_sequence(tensor_list, batch_first=True).to(device)
#
#         mask_tensor = torch.zeros_like(processed_def).to(device)
#         mask_tensor.masked_fill_(processed_def != 0, 1)
#
#         output, _ = gloss_encoder(processed_def, attention_mask=mask_tensor)
#         output = output[:, 0, :].cpu()
#         for l, t in enumerate(temp_synset_list):
#             SynsetsDict[t.name()] = output[l]
#
#     temp_synset_list = all_synsets[res_batch_idx:]
#     tensor_list = [SynsetsDefinitions[syn.name()] for syn in temp_synset_list]
#     processed_def = pad_sequence(tensor_list, batch_first=True)
#     mask_tensor = torch.zeros_like(processed_def)
#     mask_tensor.masked_fill_(processed_def != 0, 1)
#     output, _ = gloss_encoder(processed_def, mask_tensor)
#     output = output[:, 0, :]
#     for l, t in enumerate(temp_synset_list):
#         SynsetsDict[t.name()] = output[l]
#
# updateSynsetsEmbeddings()
