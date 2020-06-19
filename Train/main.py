import sys

sys.path.append('..')
from Models.ContextEncoder import contextEncoder
from Models.GlossEncoder import glossEncoder
from WSDdataloader.WSDdataLoader import get_loader
import torch
import pprint
import math
import torch.nn as nn
from transformers import BertTokenizer
from tqdm import tqdm
import argparse
import threading
from nltk.corpus import wordnet as wn
from torch.nn.utils.rnn import pad_sequence

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id1",
                    default=2,
                    type=int)
parser.add_argument('--gpu_id2')
args = parser.parse_args()

MAX_GRAD_NORM = 0.5
EPOCHS = 20
BATCH_SIZE = 2
lr_idx = 3
lr_list = [1e-6, 5e-6, 1e-5, 5e-5]

device1 = torch.device('cuda:' + str(args.gpu_id1))
device2 = torch.device('cuda:' + str(args.gpu_id2))


data_loader = get_loader(batch_size=BATCH_SIZE)  # batch_size means number of sentences
eval_loader = get_loader(batch_size=BATCH_SIZE, json_path='test.jsons')



TOTAL_STEP = len(data_loader)

context_encoder = contextEncoder(device1).to(device1)
gloss_encoder = glossEncoder(device2).to(device2)

print(gloss_encoder.model.device)


optimizer = torch.optim.Adam([{'params': context_encoder.parameters()},
                              {'params': gloss_encoder.parameters()}],
                             lr=lr_list[lr_idx])
criterion = nn.CrossEntropyLoss()


def adjust_learning_rate(optimizer, lr_idx):
    if lr_idx == 0:
        return
    else:
        lr_idx -= 1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_list[lr_idx]


def _eval():
    context_encoder.eval()
    gloss_encoder.eval()
    one_sense_word_count = 0
    correct = 0
    total = 0



    with torch.no_grad():
        for sent, sent_masks, pos_li, def_li in tqdm(data_loader):
            if len(def_li) == 0:
                continue
            sent = sent.to(device1)
            sent_masks = sent_masks.to(device1)
            context_embeddings_li = context_encoder(sent, sent_masks, pos_li)
            gloss_embeddiings_li = gloss_encoder(def_li)

            for i in range(len(context_embeddings_li)):
                sent_context = context_embeddings_li[i]
                poly_embeddings_li = gloss_embeddiings_li[i]  # [tensor1, tensor2, tensor3]

                for j in range(sent_context.size(0)):  # number of polyseme
                    total += 1
                    word_context_embedding = sent_context[j]
                    def_embeeding = poly_embeddings_li[j]  # tensor

                    scores = torch.matmul(def_embeeding, word_context_embedding) / torch.tensor(math.sqrt(768))
                    _, idx = torch.max(scores.unsqueeze(dim=0), dim=1)
                    if scores.size(0) == 1:
                        one_sense_word_count += 1

                    if idx == 0:
                        correct += 1
    print('total number: ' + str(total) + '  one sense word: ' + str(one_sense_word_count))
    return correct / total


def _train():
    context_encoder.train()
    gloss_encoder.train()

    lastAcc = -1

    for epoch in range(EPOCHS):
        for k, (sent, sent_masks, pos_li, def_li) in enumerate(data_loader):
            if len(def_li) == 0:
                continue
            sent = sent.to(device1)
            sent_masks = sent_masks.to(device1)
            context_embeddings_li = context_encoder(sent, sent_masks, pos_li)



            gloss_embeddiings_li = gloss_encoder(def_li)

            total_loss = 0
            for i in range(len(context_embeddings_li)):
                sent_context = context_embeddings_li[i]
                poly_embeddings_li = gloss_embeddiings_li[i]  # [tensor1, tensor2, tensor3]

                for j in range(sent_context.size(0)):  # number of polyseme
                    word_context_embedding = sent_context[j]
                    def_embeeding = poly_embeddings_li[j]  # tensor
                    def_embeeding = def_embeeding.to(device1)
                    scores = torch.matmul(def_embeeding, word_context_embedding) / torch.tensor(math.sqrt(768))

                    labels = torch.tensor([0]).to(device1)
                    loss = criterion(scores.unsqueeze(dim=0), labels)
                    total_loss += loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(gloss_encoder.parameters(), MAX_GRAD_NORM)
            torch.nn.utils.clip_grad_norm_(context_encoder.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            if (k + 1) % 20 == 0:
                print('Epoch: {}/{}, Step: {}/{} , loss: {:.4f}'
                      .format(epoch + 1, EPOCHS, k + 1, TOTAL_STEP, total_loss.item()))


        valid_acc = _eval()
        if lastAcc == -1:
            lastAcc = valid_acc
        else:
            if valid_acc <= lastAcc:
                adjust_learning_rate(optimizer, lr_idx=lr_idx)
        print('Epoch: {}/{} finished, val accuracy:{:.4f}'.format(epoch + 1, EPOCHS, valid_acc))



_train()
torch.save(gloss_encoder, 'gloss_encoder.pkl')
torch.save(context_encoder, 'context_encoder.pkl')
