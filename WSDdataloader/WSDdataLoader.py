from torch.utils.data import Dataset, DataLoader
from ExtractDataUtils.utils.utils import *
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
import pprint
from tqdm import tqdm
import os



if os.path.exists('SynsetsDefinitions.data'):
    SynsetsDefinitions: dict = torch.load('SynsetsDefinitions.data')
    print('SynsetsDefinitions already exists')
else:
    SynsetsDefinitions = {}
    all_synsets = list(wn.all_synsets())
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for synset in tqdm(all_synsets):
        tokenized_definitions = tokenizer.tokenize(synset.definition())
        tokenized_definitions = ['[cls]'] + tokenized_definitions
        input_definitions = tokenizer.convert_tokens_to_ids(tokenized_definitions)  # type(input_definitions) = list
        if len(input_definitions) >= 32:
            SynsetsDefinitions[synset.name()] = torch.tensor(input_definitions[:32])
        else:
            SynsetsDefinitions[synset.name()] = torch.tensor(input_definitions)
    torch.save(SynsetsDefinitions, 'SynsetsDefinitions.data')



class WSDdataset(Dataset):
    def __init__(self, json_path):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        raw_sentence_iter = iter_rawdata(json_path)

        self.processed_sentence = []
        self.processed_polyseme_positions = []
        self.processed_definitions = []

        for sentence in raw_sentence_iter:
            sentence = ' '.join(sentence)
            self.processed_sentence.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence)))

        with open(json_path) as f:
            for i, line in tqdm(enumerate(f)):
                local_polyseme_positions = []
                local_definitions = []
                sent = json.loads(line.strip())

                count = -1
                for t, word in enumerate(sent['words']):
                    original_word = word['content']
                    bert_tokenized_word = tokenizer.tokenize(original_word)
                    count += len(bert_tokenized_word)
                    if 'id' in word:
                        if len(bert_tokenized_word) == 1:
                            local_polyseme_positions.append([count])
                        else:
                            local_polyseme_positions.append(
                                list(range(count - len(bert_tokenized_word) + 1, count + 1)))

                        minimal_word_definition = []
                        answers_list = word['answers']
                        for answer in answers_list:
                            minimal_word_definition.append(wn.lemma_from_key(answer).synset().name())
                        local_definitions.append(minimal_word_definition)

                self.processed_polyseme_positions.append(local_polyseme_positions)
                self.processed_definitions.append(local_definitions)

        assert len(self.processed_sentence) == len(self.processed_polyseme_positions) == len(self.processed_definitions)

    def __getitem__(self, idx):
        return self.processed_sentence[idx], self.processed_polyseme_positions[idx], self.processed_definitions[idx]

    def __len__(self):
        return len(self.processed_sentence)





def fn(data):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    processed_sentences = []
    processed_positions = []
    processed_definitions = []

    for ids, position_list, def_list in data:
        if len(def_list) == 0:
            continue

        processed_sentences.append(torch.tensor(ids))

        processed_positions.append(position_list)



        temp_list = []
        for i, definit in enumerate(def_list):
            # definit stands for one single polyseme
            # TODO: get all the synsets of this polyseme
            right_synset = definit[0]

            # get the target polyseme
            target_pos = position_list[i]
            if len(target_pos) == 1:
                target_word_ids = ids[target_pos[0]]
            else:
                target_word_ids = ids[target_pos[0]: target_pos[-1]+1]
            if type(target_word_ids) is list:
                target_polyseme = tokenizer.decode(target_word_ids)
            else:
                target_polyseme = tokenizer.decode([target_word_ids])


            synsets = [q.name() for q in wn.synsets(target_polyseme)]

            if right_synset in synsets:
                synsets.remove(right_synset)
            synsets.insert(0, right_synset)


            for k, synset in enumerate(synsets):
                synsets[k] = SynsetsDefinitions[synset]

            synsets = pad_sequence(synsets, batch_first=True)

            synsets_mask = torch.zeros_like(synsets)
            synsets_mask.masked_fill_(synsets != 0, 1)
            tuple_processed_synsets = (synsets, synsets_mask)

            temp_list.append(tuple_processed_synsets)

        processed_definitions.append(temp_list)
    if len(processed_sentences) == 0:
        print('exceptional case!!!!!!!!!')
        return [],[],[],[]
    processed_sentences = pad_sequence(processed_sentences, batch_first=True)
    mask_tensors = torch.zeros_like(processed_sentences)
    mask_tensors.masked_fill_(processed_sentences != 0, 1)

    assert processed_sentences.size(0) == len(processed_positions) == len(processed_definitions)

    return processed_sentences, mask_tensors, processed_positions, processed_definitions







def get_loader(batch_size=64, shuffle=True, json_path='train.jsons'):
    if os.path.exists(json_path + '.data'):
        print(json_path +'dataset already exists')
        dataset = torch.load(json_path + '.data')
    else:
        print('generate '+ json_path + ' Dataset....')
        dataset = WSDdataset('../rawdata/' + json_path)
        torch.save(dataset, json_path + '.data')
        print('finish generating ' + json_path + ' Dataset')
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=fn)
    return dataloader

# if __name__ == '__main__':
    # # get_loader()
    # # Test
    # for i in range(6):
    #     print(a[i][0])
    #     print(a[i][1])
    #     print(a[i][2])
    #     print('*' * 10)
    # loader = get_loader(json_path='test.jsons')
    # for i in loader:
    #     pass

    # loader = iter(get_loader(batch_size=64))
    # a, mask, postion, def_list = next(loader)
    #
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # print(tokenizer.decode(a[0]))
    # print(postion[0][0])
    # mini_tu = def_list[0][0]
    #
    # for i in range(mini_tu[0].size(0)):
    #     print(tokenizer.decode(mini_tu[0][i]))
    # print(mini_tu[1])


    # print(c[0][0])
    # print(c[0][1])
    # print(c[1][0])
    # print(c[1][1])
    # print(c[2][0])
    # print(c[2][1])
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # print(a.size())
    # print(a)
    # print(b)
    # print(c)
    # print(tokenizer.decode(a[0]))
    # print(tokenizer.decode(a[1]))
    # print(tokenizer.decode(a[2]))
    # print(tokenizer.convert_ids_to_tokens(a[0]))
    # print(tokenizer.convert_ids_to_tokens(a[1]))
    # print(tokenizer.convert_ids_to_tokens(a[2]))
    # print(tokenizer.convert_ids_to_tokens(a[0]))
    # print(mask[0])
    # print(tokenizer.decode(tokenizer.convert_ids_to_tokens(a[1])))
    # print(mask[1])
    # print(tokenizer.convert_ids_to_tokens(a[10]))
    # print(mask[10])

    # dataset = WSDdataset('../rawdata/train.jsons')
    # # torch.save(dataset, 'dataset.data')
    # dataset: WSDdataset =
    #
    # for i in range(100, 300):
    #     sentence, pos, defi = dataset[i]
    #     print(' '.join(sentence))
    #     print(pos)
    #     print([i[0] for i in defi])
    #     print('*' * 10)
