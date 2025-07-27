## 总结
稀疏索引：BM25索引

从原理上来说，核心思想是统计文档和查询中的关键词出现频率。构建索引时，会为每个词建立一个“倒排表”，记录该词出现在哪些文档、出现了多少次。查询时，将用户输入的关键词在倒排表中查找，统计每个文档与查询的相关性分数（BM25打分公式），返回分数最高的文档。

稀疏向量：每个文档用一个高维、稀疏的词袋向量表示（大部分维度为0）。

倒排索引结构使得检索速度极快，适合大规模文本库。

什么是倒排表：每个词对应一个“倒排链表”，链表中存储了该词出现的所有文档ID及相关信息。多关键词检索时，只需对多个倒排链表做交集、并集等集合操作

正排表：文档 → 词

只能基于词面匹配，无法理解同义词、上下文等深层语义

稠密索引：E5-Flat，E5-HNSW64

E5是一种预训练的语义检索模型（如Transformer），能将文本编码为低维、稠密的向量（embedding）。构建索引时，先用E5模型将所有文档编码为向量，然后将这些向量存入向量数据库（如FAISS）。查询时，将用户输入编码为向量，计算与所有文档向量的相似度（如余弦相似度），返回最相似的文档。

稠密向量：每个文档用一个低维、稠密的浮点向量表示

支持语义检索：能理解同义词、上下文、语义相似等复杂关系。

在线搜索引擎：Google Search API

## Search Engine

In this document, we provide examples of how to launch different retrievers, including local sparse retriever (e.g., BM25), local dense retriever (e.g., e5) and online search engine.
For local retrievers, we use [wiki-18](https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus) corpus as an example and the corpus indexing can be found at [bm25](https://huggingface.co/datasets/PeterJinGo/wiki-18-bm25-index), [e5-flat](https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index), [e5-HNSW64](https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index-HNSW64).

### How to choose the retriever?

- If you have a private or domain-specific corpus, choose **local retriever**本地检索器

    - If there is no high quality embedding-based retrievers (dense retrievers) in your domain, choose **sparse local retriever** (e.g., BM25).稀疏检索器（如BM25）适用于没有高质量嵌入检索器的领域。检索效率高，不需要GPU

    - Otherwise choose **dense local retriever**.密集检索器（如E5）
    
        - If you do not have sufficent GPUs to conduct exact dense embedding matching, choose **ANN indexing** on CPUs. ANN索引：CPU上进行近似匹配，快速但可能不够准确

        - If you have sufficient GPUs, choose **flat indexing** on GPUs.Flat索引：GPU上进行精确匹配，准确但较慢


- If you want to train a general LLM search agent and have enough funding, choose **online search engine** (e.g., [SerpAPI](https://serpapi.com/)).在线搜索引擎，适用于训练通用LLM搜索代理


- If you have a domain specific online search engine (e.g., PubMed search), you can refer to [link](https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/serp_search_server.py) to integrate it to Search-R1 by yourself.推荐使用SerpAPI（集成多个搜索引擎，无月度配额限制）

Search engine launching scripts can be found at [link](https://github.com/PeterGriffinJin/Search-R1/tree/main/example/retriever).

### Local Sparse Retriever

本地稀疏检索器（BM25）

Sparse retriever (e.g., bm25) is a traditional method. The retrieval process is very efficient and no GPUs are needed. However, it may not be as accurate as dense retrievers in some specific domain.

(1) Download the indexing.下载预构建索引
```bash
save_path=/your/path/to/save
huggingface-cli download PeterJinGo/wiki-18-bm25-index --repo-type dataset --local-dir $save_path
```

(2) Launch a local BM25 retriever server.
```bash
conda activate retriever

index_file=$save_path/bm25
corpus_file=$save_path/wiki-18.jsonl
retriever_name=bm25

python search_r1/search/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name
```


### Local Dense Retriever

You can also adopt some off-the-shelf dense retrievers, e.g., e5. These models are much stronger than sparse retriever in some specific domains.
If you have sufficient GPU, we would recommend the flat indexing variant below, otherwise you can adopt the ANN variant.

#### Flat indexing

Flat indexing conducts exact embedding match, which is slow but very accurate. To make it efficient enough to support online RL, we would recommend enable **GPU** usage by ```--faiss_gpu```.

(1) Download the indexing and corpus.
```bash
save_path=/the/path/to/save
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

(2) Launch a local flat e5 retriever server.

```bash
conda activate retriever

index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python search_r1/search/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name --retriever_model $retriever_path --faiss_gpu

```


#### ANN indexing (HNSW64)

To improve the search efficient with only **CPU**, you can adopt approximate nearest neighbor (ANN) indexing, e.g., with HNSW64.
It is very efficient, but may not be as accurate as flat indexing, especially when the number of retrieved passages is small.

(1) Download the indexing.
```bash
save_path=/the/path/to/save
huggingface-cli download PeterJinGo/wiki-18-e5-index-HNSW64 --repo-type dataset --local-dir $save_path
cat $save_path/part_* > $save_path/e5_HNSW64.index
```


(2) Launch a local ANN dense retriever server.
```bash
conda activate retriever

index_file=$save_path/e5_HNSW64.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python search_r1/search/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name --retriever_model $retriever_path
```


### Online Search Engine

We support both [Google Search API](https://developers.google.com/custom-search/v1/overview) and [SerpAPI](https://serpapi.com/). We would recommend [SerpAPI](https://serpapi.com/) since it integrates multiple online search engine APIs (including Google, Bing, Baidu, etc) and does not have a monthly quota limitation ([Google Search API](https://developers.google.com/custom-search/v1/overview) has a hard 10k monthly quota, which is not sufficient to fulfill online LLM RL training).

#### SerAPI online search server

```bash
search_url=https://serpapi.com/search
serp_api_key="" # put your serp api key here (https://serpapi.com/)

python search_r1/search/serp_search_server.py --search_url $search_url --topk 3 --serp_api_key $serp_api_key
```

#### Google online search server

```bash
api_key="" # put your google custom API key here (https://developers.google.com/custom-search/v1/overview)
cse_id="" # put your google cse API key here (https://developers.google.com/custom-search/v1/overview)

python search_r1/search/google_search_server.py --api_key $api_key --topk 5 --cse_id $cse_id --snippet_only
```

