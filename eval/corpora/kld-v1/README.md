# NInfer KLD Corpus

`ninfer-kld-v1` is the single-stream corpus for KL-divergence comparisons between a reference model
and quantized candidates (see [Logits dumps and KL divergence](../../../docs/perplexity.md#logits-dumps-and-kl-divergence)).
`llama-perplexity` scores one text file, so the sources are concatenated into one stream and every
compared run evaluates the same token ids.

| Source | Domain | Token budget |
|---|---|---:|
| `perplexity-1m/data/wikitext/00.txt` | `english_reference` | 28,672 |
| `perplexity-1m/data/pg19/00.txt` | `english_long_form` | 28,672 |
| `perplexity-1m/data/zhwiki/00.txt` | `chinese_reference` | 28,672 |
| `data/french/00.txt` | `french` | 6,144 |
| `data/chat/00.txt` | `chat_reasoning` | 6,144 |
| `perplexity-1m/data/ninfer/00.txt` | `ninfer_code` | remainder |

`manifest.json` fixes the order and budgets for 32 windows of 4,096 tokens, split into four
interleaved rounds: each round takes a quarter of every budget, so the first 8 windows are a
proportional sample of all domains. `tools/kld/build_corpus.py` cuts each source at the first line
boundary that reaches its per-round budget, joins the pieces with a blank line, and records the
resulting token range of every piece. Budgets are in reference-tokenizer
tokens; the built corpus and its `tokens.bin` are outputs, not committed data.

The French text (essays, a short story, a letter, a recipe, technical explanations, a dialogue) and
the chat/reasoning transcript (user questions with worked, checked answers on arithmetic, logic,
code, SQL, science and planning) were written for this corpus and are distributed under the
repository license. They contain no chat-template or tool-call markers: those parse as special
tokens in the NInfer tokenizer but as plain text in `llama-perplexity`. The other sources and their
notices are those of [`ninfer-ppl-1m-v1`](../perplexity-1m/README.md).
