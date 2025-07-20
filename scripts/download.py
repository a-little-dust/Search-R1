import argparse
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser(description="Download files from a Hugging Face dataset repository.")
# 仓库ID
parser.add_argument("--repo_id", type=str, default="PeterJinGo/wiki-18-e5-index", help="Hugging Face repository ID")
# 保存路径
parser.add_argument("--save_path", type=str, required=True, help="Local directory to save files")
    
args = parser.parse_args()

repo_id = "PeterJinGo/wiki-18-e5-index"  # E5索引仓库
# 它支持Flat索引（精确匹配，需要GPU）和HNSW64索引（近似匹配，速度快，但精度略低）
for file in ["part_aa", "part_ab"]:  # 下载两个分片文件"part_aa"和"part_ab"
    hf_hub_download(
        repo_id=repo_id,
        filename=file,  # e.g., "e5_Flat.index"
        repo_type="dataset",
        local_dir=args.save_path,
    )

# wiki-18语料库仓库，提供检索系统的文档源，包含18个维基百科类别的文档，每行一个JSON对象，包含文档标题和内容
repo_id = "PeterJinGo/wiki-18-corpus"
hf_hub_download(
        repo_id=repo_id,
        filename="wiki-18.jsonl.gz",
        repo_type="dataset",
        local_dir=args.save_path,
)
