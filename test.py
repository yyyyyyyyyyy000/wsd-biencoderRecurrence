# # # from transformers import BertTokenizer
# # # import pprint
# # # from nltk.corpus import wordnet as wn
# # # import torch
# # # # cased = BertTokenizer.from_pretrained('bert-base-uncased')
# # # # print(cased.vocab)
# # # # a = cased.vocab
# # # # print(a['beautiful'])
# # # # print(a['Beautiful'])
# # #
# # # #
# # # # a = wn.synsets('car')
# # # # b = wn.synset('car.n.01')
# # # # for i in b.lemmas():
# # # #     print(i.name())
# # # # print(b.examples())
# # # # print(b.name())
# # # # a = wn.lemma_from_key('simple%5:00:00:retarded:00').synset().name()
# # # # print(a)
# # # # b = wn.lemma_from_key("effort%1:04:00::").synset().name()
# # # # print(b)
# # # #
# # # # synseta = wn.synset(a)
# # # # synsetb = wn.synset(b)
# # # # print(synseta.definition())
# # # # print(synsetb.definition())
# # #
# # # # a = wn.synset('transport.v.02')
# # # # print(a.definition())
# # # # lemmas = a.lemmas()
# # # # print(lemmas)
# # # # for lemma in lemmas:
# # # #     print(lemma.key())
# # # import torch
# # # # a = torch.zeros(3,5)
# # # # mask = (a != 0)
# # # # print(mask)
# # # # a = torch.zeros(1, 3)
# # # # b = torch.ones(1, 3)
# # # # c = torch.cat((a, b), dim=0)
# # # # c = torch.cat((c, a), dim=0)
# # # # print(c)
# # # # b = torch.ones(3)
# # # # c = torch.stack((a, b), dim=0)
# # # # c = torch.stack((c, a), dim=0)
# # # # print(c)
# # # # print(a)
# # # # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# # # # # st = 'he plant beautifullyhhhhasdasd asdasdaa'
# # # # a = tokenizer.tokenize(st)
# # # # # print(a)
# # # # b = tokenizer.decode(5136)
# # # # print(b)
# # # # a = [1,2,3]
# # # # print(type(a) is list)
# # # # a = [1,2,3,4,5,6,7,8]
# # # # b = a[3:5]
# # # # print(b)
# # # # a = wn.synsets('beautiful')
# # # # print(a)
# # # # a.remove(wn.synset('beautiful.a.01'))
# # # # print(a)
# # # # a.insert(0,wn.synset('beautiful.a.01'),)
# # # # # print(a)
# # # # a = wn.synsets('ever')
# # # # for t in a:
# # # #     print(t.definition())
# # # # a = torch.arange(0,12).reshape(3,2,2)
# # # # print(a)
# # # # print(a[:, 0, :])
# # # # a = torch.randn(3, 5)
# # # # b = torch.tensor([1,1,1,1,1],).float()
# # # # print(torch.matmul(a, b))
# # # # a = torch.zeros(4)
# # # # a[0] = 1
# # # # print(a)
# # # # labels = torch.tensor([0])
# # # # print(labels)
# # # # a = torch.tensor(28)
# # # # print(a)
# # #
# # # # a = torch.tensor([[1.5, 2.1, 1.4, 1.3]]).float()
# # # # b = torch.softmax(a, dim=1)
# # # # print(b)
# # # #
# # # # # b = torch.softmax(b, dim=1)
# # # # # print(b)
# # # # label = torch.tensor([0])
# # # # criterion = torch.nn.CrossEntropyLoss()
# # # # los = criterion(a, label)
# # # # print(los)
# # # a = torch.tensor([[112312,3,4,51,6]])
# # # _, idx = torch.max(a,dim=1)
# # # print(idx)
# # # if idx == 0:
# # #     print('asd')
# # # else:
# # #     print('qqqqq')
# # # # b = 0
# # # # c = a+b
# # # # c = c.unsqueeze(dim=0)
# # # # print(c)
# # # print(torch.tensor(1)/torch.tensor(3))
# # import torch
# # # a = torch.randn(3,5)
# # # print(a[2,3,4])
# # # from nltk.corpus import wordnet as wn
# # #
# # # a = wn.synsets('preposterous')
# # # print(a[0].definition())
# #
# #
# # import threading
# # import sys
# # import time
# # # def run(tt):
# # #     global mutex
# # #     for i in range(0, 10000):
# # #         tt.append(3)
# # #         time.sleep(0.001)
# # #     mutex.release()
# #
# #
# # # mutex = threading.Lock()
# # # tt = [123,3,13,1,]
# # # t1 = threading.Thread(target=run, args=(tt,))
# # # t1.start()
# # # while len(tt) == 4:
# # #     continue
# # # mutex.acquire()
# # # print(tt)
# # # a = [1,2,3,4,5]
# # # def run():
# # #     global a
# # #     a = [1,2,3,4]
# # #
# # # run()
# # # print(a)
# # #
# # # a = 1.0
# # # print(int(a))
# # # from nltk.corpus import wordnet as wn
# # # a = list(wn.all_synsets())
# # # print(len(a))
# # from transformers import BertModel, BertTokenizer
# #
# #
# #
# # a = torch.rand(64, 64) * 1000
# # a.floor_()
# # a = a.long()
# #
# #
# #
# # device = torch.device('cpu')
# # a = a.to(device)
# # # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased').to(device)
# # model = BertModel.from_pretrained('bert-base-uncased', from_tf=True).to(device)
# # # b = 'asdasd asd asd sa asd '
# # # print(tokenizer.tokenize(b))
# # count = 0
# # # while True:
# # #     output = model(a)
# # #
# # #
# # #     print('finish: ', count+1)
# # #     count += 1
# #
# from nltk.corpus import wordnet as wn
# # a = wn.synsets('many')
# # print(a)
# from transformers import BertTokenizer, BertModel
# # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# # model = BertModel.from_pretrained('bert-base-uncased', from_tf=True)
# # import torch
# # # print(tokenizer.decode([[5532, 2312,1111,2222],[1123,1232,1112,3333,2222]]))
# # # a = list(wn.all_synsets())
# # a = ['he', 'plant', 'a', 'tree']
# # b = tokenizer.convert_tokens_to_ids(a)
# # print(b)
# # import torch
# # print(model(torch.tensor([[1231,123,2312,1123]])))
# import time
# import  torch
#
#
# SynsetsDict = {}
# SynsetsDefinitions = {}
# #
# # all_synsets = list(wn.all_synsets())
# # from tqdm import tqdm
# # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# # start = time.clock()
# # for synset in tqdm(all_synsets):
# #     tokenized_definitions = tokenizer.tokenize(synset.definition())
# #     tokenized_definitions = ['[cls]'] + tokenized_definitions
# #     input_definitions = tokenizer.convert_tokens_to_ids(tokenized_definitions)  # type(input_definitions) = list
# #     model(torch.tensor([input_definitions]))
# #     SynsetsDefinitions[synset.name()] = input_definitions
# # end = time.clock()
# # print(str(end-start))
# from torch.nn.utils.rnn import pad_sequence
# a = pad_sequence([torch.tensor([1,2,3]),torch.tensor([1,2,3,4,5,]),torch.tensor([1,2,3,4,5,6,7,1,1])],batch_first=True)
# print(a)
#
#
from transformers import BertModel
from torch.nn.utils.rnn import pad_sequence
import torch
device = torch.device('cuda: 1')
encoder = BertModel.from_pretrained('bert-base-uncased', from_tf=True).to(device)
decoder = BertModel.from_pretrained('bert-base-uncased', from_tf=True).to(device)
a = torch.rand(4, 64) * 100
b = torch.ones(16, 64).long().to(device)
a = a.floor().long().to(device)

while True:

    output1 = encoder(a)
    output2 = decoder(b)
#
#
#
#
