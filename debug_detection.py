from run_pipeline_v2 import is_vietnamese_word, clean_text_content

# Simulating the pipeline logic
ambiguous_set = {"do", "to", "on", "a", "an"}

def resolve_context(seg_words):
    # 2. Resolve Ambiguous Words based on Bidirectional Context
    for k, w_item in enumerate(seg_words):
        if w_item["is_ambiguous"]:
            prev_is_vn = None
            if k > 0: prev_is_vn = seg_words[k-1]["is_vn"]
            
            next_is_vn = None
            if k < len(seg_words) - 1: next_is_vn = seg_words[k+1]["is_vn"]
            
            if (prev_is_vn is False) or (next_is_vn is False):
                w_item["is_vn"] = False
            elif prev_is_vn is not None:
                    w_item["is_vn"] = prev_is_vn
            elif next_is_vn is not None:
                    w_item["is_vn"] = next_is_vn
            else:
                w_item["is_vn"] = False
    return seg_words

def test_case(text_list):
    seg_words = []
    for raw_w in text_list:
        clean_w = clean_text_content(raw_w)
        is_vn = is_vietnamese_word(raw_w)
        lower_w = clean_w.lower()
        is_ambiguous = (lower_w in ambiguous_set) or (lower_w.isdigit() and len(lower_w) == 1)
        seg_words.append({"text": clean_w, "is_vn": is_vn, "is_ambiguous": is_ambiguous})
    
    resolved = resolve_context(seg_words)
    labels = []
    for item in resolved:
        lbl = "[vi]" if item["is_vn"] else "[en]"
        labels.append(f"{lbl}{item['text']}")
    print(" ".join(labels))


