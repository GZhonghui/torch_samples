from transformers import AutoTokenizer


def show_tokenization(tokenizer, text: str) -> None:
    # 人类语言 -> 切分为token（这一步还是字符串，是一个字符串列表） -> 整数ID（数字列表）
    # 但其实中间这个token的字符串列表，我们不太关心。

    # tokenizer 接收原始自然语言字符串，先做规范化与子词切分，再映射成整数 ID。
    # return_tensors="pt" 会把结果直接组织为 PyTorch Tensor。
    # 也就是：人类语言 ---(直接到ID)--> token ids (tensor)
    encoded = tokenizer(text, return_tensors="pt")

    # encoded 可以理解为“分词后的结果包”，常见类型是 BatchEncoding（行为类似 dict）。
    # 里面会按键存放 input_ids / attention_mask 等张量。
    encoded_type = type(encoded).__name__
    encoded_keys = list(encoded.keys())

    # 这里的 input_ids shape 是 [batch_size, seq_len]。
    # seq_len 是分词后 token 的数量，通常和原本的文本长度不完全一样，因为人类语言的单词和token不是一一对应的。
    # 本示例每次只喂 1 句话，所以 batch_size = 1。
    input_shape = tuple(encoded["input_ids"].shape)
    mask_shape = tuple(encoded["attention_mask"].shape)

    # 为了可读性，把第 0 条样本从 Tensor 转成 Python list。
    # 这里取[0]就是取得 batch 中的第一条数据，tolist() 则把 Tensor 转成普通的 Python list。
    input_ids = encoded["input_ids"][0].tolist()

    # id -> token：把整数 ID 还原成对应的子词单元。
    # 这里的 tokens 是字符串的列表，每个字符串是一个 token（可能是一个完整单词，也可能是一个子词）。
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # 再做一次 decode，验证 token/id 是否能重建为近似原文本。
    # 注意这里decode用的是ids，不是tokens。
    decoded = tokenizer.decode(input_ids, skip_special_tokens=True)

    print("=" * 80)
    print(f"Original text : {text}")
    print(f"Input text length   : {len(text)} characters")
    print(f"encoded type        : {encoded_type}")
    print(f"encoded keys        : {encoded_keys}")
    print(f"input_ids shape     : {input_shape}")
    print(f"attention_mask shape: {mask_shape}")
    print(f"Tokens        : {tokens}")
    print(f"Token IDs     : {input_ids}")
    print(f"Attention mask: {encoded['attention_mask'][0].tolist()}")
    print(f"Decoded text  : {decoded}")


def main() -> None:
    # 改为本地加载：只依赖你已经下载到磁盘的 tokenizer 文件。
    # 目录中包含 vocab.json / merges.txt / tokenizer_config.json 等。
    model_path = "/home/zh-ge/models/opt-13b/"

    # tokenizer 是基于特定模型的，每个模型都有自己独特的分词规则和词表。加载时需要指定模型路径。
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    print(f"Tokenizer path: {model_path}")
    print("下面演示自然语言如何被 Transformer tokenizer 转成 token 与 token id。\n")

    examples = [
        "今天天气不错，我们去公园散步吧！",
        "Transformer breaks words into subword tokens for modeling.",
        "tokenization helps models understand unbelievable words.",
    ]

    for text in examples:
        show_tokenization(tokenizer, text)


if __name__ == "__main__":
    main()
