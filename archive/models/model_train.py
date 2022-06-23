import torch
from keybert import KeyBERT
kw_model = KeyBERT()

def replace_with_entities(parent_dict):
    if 'kids' in parent_dict.keys():
        for i in range(len(parent_dict['kid_texts'])):
            if 'text_proc' in parent_dict['kid_texts'][i]:
                doc = parent_dict["kid_texts"][i]['text_proc']
                parent_dict["kid_texts"][i]['text_proc'] = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), stop_words='english')
                replace_with_entities(parent_dict["kid_texts"][i])
                
def get_text_pairs(data_map, entity_map, n_entities, entity_idx):
    all_examples = []
    if "kid_texts" in data_map:
        for child in range(len(data_map["kid_texts"])):
            if "text_proc" in data_map["kid_texts"][child] and "kid_texts" in data_map["kid_texts"][child]:
                for gc in range(len(data_map["kid_texts"][child]["kid_texts"])):
                    if "text_proc" in entity_map["kid_texts"][child]["kid_texts"][gc] and "kid_texts" in data_map["kid_texts"][child]["kid_texts"][gc]:
                        for ggc in range(len(data_map["kid_texts"][child]["kid_texts"][gc]["kid_texts"])):
                            if "text_proc" in data_map["kid_texts"][child]["kid_texts"][gc]["kid_texts"][ggc]:
                                first_post = data_map["kid_texts"][child]["text_proc"]
                                entity_dist = entity_map["kid_texts"][child]["kid_texts"][gc]["text_proc"]
                                ground_truth = torch.zeros(n_entities)
                                for entity, prob in entity_dist:
                                    ground_truth[entity_idx[entity]] = prob
                                last_post = data_map["kid_texts"][child]["kid_texts"][gc]["kid_texts"][ggc]["text_proc"]
                                all_examples.append((first_post, last_post, ground_truth))
            all_examples += get_text_pairs(data_map["kid_texts"][child], entity_map["kid_texts"][child], n_entities, entity_idx)
    return all_examples

def get_pretrained_model_tokenizer(nclasses):
    model = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', 'bert-base-cased-finetuned-mrpc')
    model.classifier = torch.nn.Linear(768, nclasses)
    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased-finetuned-mrpc')
    return tokenizer, model

def run_model(str1s, str2s, tokenizer, model):
    tokens = []
    ids = []
    for i in range(len(str1s)):
        tokens.append(tokenizer.encode(str1s[i], str2s[i], add_special_tokens=True))
        s2 = len(tokenizer.encode(str2s[i], add_special_tokens=True))
        ids = ([0] * (len(tokens[i]) - s2)) + ([1] * s2)
    tokens = torch.tensor(tokens)
    ids = torch.tensor(ids)
    return model(tokens, token_type_ids=ids)[0]

def entity_count_map_rec(data_map):
    entity_map = {}
    count = 0
    if "text_proc" in data_map:
        for entity in data_map["text_proc"]:
            entity_map[entity[0]] = count
            count += 1
    if "kid_texts" in data_map:
        for child in data_map["kid_texts"]:
            count = entity_count_map_rec_adder(child, entity_map, count)
    return entity_map, count

def entity_count_map_rec_adder(data_map, entity_map, current_count):
    count = current_count
    if "text_proc" in data_map:
        for entity in data_map["text_proc"]:
            if entity[0] not in entity_map:
                entity_map[entity[0]] = count
                count += 1
    if "kid_texts" in data_map:
        for child in data_map["kid_texts"]:
            count = entity_count_map_rec_adder(child, entity_map, count)
    return count

def main_loop(texts, tokenizer, model, optim, batch_size = 1, epochs = 1):
    loss_f = torch.nn.KLDivLoss()
    for epoch in range(epochs):
        current_batch_s1 = []
        current_batch_s2 = []
        current_batch_gt = []
        count = 0
        for text in texts:
            current_batch_s1.append(text[0])
            current_batch_s2.append(text[1])
            current_batch_gt.append(text[2])
            count += 1
            if count % batch_size == 0:
                optim.zero_grad()
                output = run_model(current_batch_s1, current_batch_s2, tokenizer, model)
                loss = loss_f(output, torch.stack(current_batch_gt))
                loss.backward()
                optim.step()
                current_batch_s1 = []
                current_batch_s2 = []
                current_batch_gt = []
        if count % batch_size != 0:
            optim.zero_grad()
            output = run_model(current_batch_s1, current_batch_s2, tokenizer, model)
            loss = loss_f(output, torch.stack(current_batch_gt))
            loss.backward()
            optim.step()
            current_batch_s1 = []
            current_batch_s2 = []
            current_batch_gt = []