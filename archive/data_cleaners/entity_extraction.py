idn = 29218144
parent_dict = recursive_lookup(idn)

def replace_with_entities(parent_dict):
    if 'kids' in parent_dict.keys():
        for i in range(len(parent_dict['kid_texts'])):
            if 'text_proc' in parent_dict['kid_texts'][i]:
                doc = parent_dict["kid_texts"][i]['text_proc']
                parent_dict["kid_texts"][i]['text_proc'] = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), stop_words='english')
                replace_with_entities(parent_dict["kid_texts"][i])