# RAG-Anything → LightRAG 完整数据流

本文档详细描述了文档从 RAG-Anything 解析到 LightRAG 存储和处理的完整数据流程。

---

## 数据流概览

```
用户上传 → RAG-Anything 解析 → 分离内容 → LightRAG 插入 → 切片处理 → 向量化 → 知识图谱提取
```

---

## T0: 用户上传文档

```
┌─────────────────────────────────────────┐
│  用户上传文档                            │
│  └─ document.pdf (10 pages, 50KB)      │
└─────────────────────────────────────────┘
```

**输入**: 原始文档文件（PDF, Word, PPT, 图片等）

---

## T1: RAG-Anything 解析文档

```python
# parse_document() 调用
MineruParser.parse()
  ├─ 调用 mineru CLI
  ├─ 生成 content_list.json (结构化数据)
  └─ 生成 document.md (副产品)
```

### 输出: content_list.json

```json
[
  {
    "type": "text",
    "text": "Intro paragraph..."
  },
  {
    "type": "image",
    "img_path": "img1.png",
    "img_caption": "Figure 1: Architecture"
  },
  {
    "type": "text",
    "text": "Main content..."
  },
  {
    "type": "table",
    "table_body": [
      ["Header1", "Header2"],
      ["Data1", "Data2"]
    ]
  }
]
```

**关键点**:
- MinerU 将文档解析为结构化的内容列表
- 每个元素都有明确的类型标记
- 保留了文档的原始结构信息

---

## T2: RAG-Anything 分离内容

```python
# utils.separate_content(content_list)
def separate_content(content_list):
    text_blocks = []
    multimodal_items = []

    for item in content_list:
        if item["type"] == "text":
            text_blocks.append(item["text"])
        else:
            multimodal_items.append(item)

    # 用双换行连接所有文本块
    text_content = "\n\n".join(text_blocks)

    return text_content, multimodal_items
```

### 输出

**text_content** (纯文本字符串):
```
Intro paragraph...

Main content...

More text content continues here...
```

**multimodal_items** (多模态内容列表):
```python
[
    {"type": "image", "img_path": "img1.png", ...},
    {"type": "table", "table_body": [[...]]},
    {"type": "equation", "latex": "E=mc^2"}
]
```

**关键点**:
- 所有文本块被合并成一个完整字符串
- 多模态内容单独保存
- **此时尚未进行任何切片操作**

---

## T3: RAG-Anything 调用 LightRAG 插入文本

```python
# raganything/api/raganything_server.py
async def insert_text_content(
    lightrag: LightRAG,
    text_content: str,
    doc_id: str,
    file_name: str
):
    """将完整文本内容插入 LightRAG"""
    await lightrag.ainsert(
        input=text_content,      # ← 完整的纯文本，未切片
        file_paths=file_name,    # ← 原始文件名
        ids=doc_id               # ← 文档 ID
    )
```

### 传递给 LightRAG 的数据

```python
{
    "input": "Intro paragraph...\n\nMain content...\n\n...",  # 完整文本，可能数千到数万字符
    "file_paths": "document.pdf",
    "ids": "doc-abc123"
}
```

**关键点**:
- **传递的是完整的、未切片的文本字符串**
- RAG-Anything 不负责文本切片
- 切片工作完全由 LightRAG 内部完成

---

## ━━━ LightRAG 边界 ━━━

从此处开始，所有处理都在 LightRAG 内部进行。

---

## T4: LightRAG 接收并入队文档

```python
# lightrag/lightrag.py
async def ainsert(
    self,
    input: str,
    ids: str,
    file_paths: str
):
    """接收文档并加入处理队列"""

    # 1. 存储完整文档到 KV Storage
    await self.full_docs.upsert({
        ids: {
            "content": input,           # ← 完整文本
            "file_path": file_paths,
            "created_at": datetime.now()
        }
    })

    # 2. 创建文档状态记录
    await self.doc_status.upsert({
        ids: {
            "status": "PENDING",
            "chunks_count": 0,
            "created_at": datetime.now()
        }
    })

    # 3. 加入处理队列
    await self.pipeline_enqueue_documents(ids)
```

### 存储状态

**KV Storage (full_docs)**:
```json
{
  "doc-abc123": {
    "content": "Intro paragraph...\n\nMain content...",
    "file_path": "document.pdf",
    "created_at": "2025-10-03T12:00:00"
  }
}
```

**Doc Status**:
```json
{
  "doc-abc123": {
    "status": "PENDING",
    "chunks_count": 0
  }
}
```

**关键点**:
- 文档以完整形式存储在 `full_docs`
- 状态标记为 `PENDING`，等待处理
- **此时仍未进行切片**

---

## T5: LightRAG 处理管道启动

```python
# lightrag/kg/shared_storage.py
async def pipeline_process_enqueue_documents(rag: LightRAG):
    """后台任务：处理待处理文档队列"""

    while True:
        # 1. 从队列获取待处理文档
        doc_ids = await get_pending_documents()

        # 2. 并发处理文档
        for doc_id in doc_ids:
            await process_document(rag, doc_id)

        await asyncio.sleep(1)
```

**关键点**:
- 这是一个后台异步任务
- 从队列中拉取 `PENDING` 状态的文档
- 调用 `process_document()` 进行实际处理

---

## T6: ✅ LightRAG 执行切片（核心步骤！）

```python
# lightrag/lightrag.py:1547
async def process_document(self, doc_id: str):
    """处理单个文档：切片、向量化、实体提取"""

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤 1: 读取完整文本
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    doc_data = await self.full_docs.get_by_id(doc_id)
    content = doc_data["content"]
    # content = "Intro paragraph...\n\nMain content..."
    # 这是一个完整的、未切片的字符串

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤 2: ✅ 执行切片算法（关键步骤！）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    chunks = {
        chunk_id: chunk_data
        for dp in self.chunking_func(
            tokenizer=self.tokenizer,
            content=content,              # ← 传入完整文本
            max_token_size=1024,          # ← 每个切片的最大 token 数
            overlap_token_size=128        # ← 切片间的重叠 token 数
        )
        for chunk_id, chunk_data in [(dp["chunk_id"], dp)]
    }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤 3: 切片算法详解
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 见下方详细说明

    return chunks
```

### 切片算法详解

```python
# lightrag/operate.py:66
def chunking_by_token_size(
    tokenizer,
    content: str,
    max_token_size: int = 1024,
    overlap_token_size: int = 128
):
    """基于 token 数量的滑动窗口切片算法"""

    # 1. 将文本编码为 tokens
    tokens = tokenizer.encode(content)
    # tokens = [101, 2023, 2003, 1037, ..., 102]
    # 假设 total_tokens = 5000

    results = []
    index = 0

    # 2. 滑动窗口切片
    # stride = max_token_size - overlap_token_size
    # stride = 1024 - 128 = 896
    for start in range(0, len(tokens), max_token_size - overlap_token_size):
        # 提取当前窗口的 tokens
        end = min(start + max_token_size, len(tokens))
        chunk_tokens = tokens[start:end]

        # 解码为文本
        chunk_text = tokenizer.decode(chunk_tokens)

        # 生成切片 ID
        chunk_id = compute_mdhash_id(chunk_text, prefix="chunk-")

        # 保存切片数据
        results.append({
            "chunk_id": chunk_id,
            "content": chunk_text,
            "tokens": len(chunk_tokens),
            "chunk_order_index": index,
            "full_doc_id": doc_id
        })

        index += 1

    return results
```

### 切片示例

假设输入文本经过 tokenization 后有 **5000 tokens**，配置如下：
- `max_token_size = 1024`
- `overlap_token_size = 128`
- `stride = 896`

**切片过程**:

| 切片索引 | Token 范围 | Token 数量 | 文本内容示例 |
|---------|-----------|-----------|-------------|
| 0 | 0-1024 | 1024 | "Intro paragraph..." |
| 1 | 896-1920 | 1024 | "...overlap...Main content..." |
| 2 | 1792-2816 | 1024 | "...overlap...Section 2..." |
| 3 | 2688-3712 | 1024 | "...overlap...Section 3..." |
| 4 | 3584-4608 | 1024 | "...overlap...Section 4..." |
| 5 | 4480-5000 | 520 | "...overlap...Conclusion" |

**总切片数**: 6 个切片

### 切片结果数据结构

```python
chunks = {
    "chunk-abc123": {
        "content": "Intro paragraph with detailed explanation...",
        "tokens": 1024,
        "chunk_order_index": 0,
        "full_doc_id": "doc-xyz789"
    },
    "chunk-def456": {
        "content": "...overlap text...Main content continues with more details...",
        "tokens": 1024,
        "chunk_order_index": 1,
        "full_doc_id": "doc-xyz789"
    },
    "chunk-ghi789": {
        "content": "...overlap text...Section 2 introduces new concepts...",
        "tokens": 1024,
        "chunk_order_index": 2,
        "full_doc_id": "doc-xyz789"
    },
    # ... 更多切片
    "chunk-xyz999": {
        "content": "...overlap text...Conclusion summarizes the findings.",
        "tokens": 520,
        "chunk_order_index": 5,
        "full_doc_id": "doc-xyz789"
    }
}
```

**关键点**:
- ✅ **切片在 LightRAG 内部完成**
- ✅ **使用滑动窗口算法**，相邻切片有重叠
- ✅ 每个切片保留完整的语义上下文
- ✅ 切片数量 = `⌈total_tokens / stride⌉`

---

## T7: LightRAG 存储切片

```python
# lightrag/lightrag.py
async def process_document(self, doc_id: str):
    # ... 切片完成后 ...

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤 1: 更新文档状态
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    await self.doc_status.upsert({
        doc_id: {
            "status": "PROCESSING",
            "chunks_count": len(chunks),     # ← 切片总数
            "chunks_list": list(chunks.keys())  # ← 切片 ID 列表
        }
    })

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤 2: 存储切片文本到 KV Storage
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    await self.text_chunks.upsert(chunks)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤 3: 向量化并存储到 Vector DB
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    embeddings = await self.embedding_func(
        [chunk["content"] for chunk in chunks.values()]
    )

    await self.chunks_vdb.upsert({
        chunk_id: {
            "content": chunk["content"],
            "embedding": emb
        }
        for chunk_id, chunk, emb in zip(chunks.keys(), chunks.values(), embeddings)
    })
```

### 存储位置

**1. KV Storage (text_chunks)**:
```json
{
  "chunk-abc123": {
    "content": "Intro paragraph...",
    "tokens": 1024,
    "chunk_order_index": 0,
    "full_doc_id": "doc-xyz789"
  },
  "chunk-def456": { ... }
}
```

**2. Vector Storage (chunks_vdb)**:
```json
{
  "chunk-abc123": {
    "content": "Intro paragraph...",
    "embedding": [0.123, -0.456, 0.789, ...]  // 768维或1024维向量
  },
  "chunk-def456": { ... }
}
```

**3. Doc Status**:
```json
{
  "doc-xyz789": {
    "status": "PROCESSING",
    "chunks_count": 6,
    "chunks_list": [
      "chunk-abc123",
      "chunk-def456",
      "chunk-ghi789",
      "chunk-jkl012",
      "chunk-mno345",
      "chunk-xyz999"
    ]
  }
}
```

---

## T8: LightRAG 提取实体和关系

```python
# lightrag/lightrag.py
async def process_document(self, doc_id: str):
    # ... 切片存储完成后 ...

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤 1: 实体提取
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    entities = await self._process_extract_entities(chunks)
    # 对每个切片调用 LLM 提取实体
    # 示例输出:
    # {
    #   "entity-001": {"name": "RAG System", "type": "Technology"},
    #   "entity-002": {"name": "Knowledge Graph", "type": "Concept"}
    # }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤 2: 关系提取
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    relations = await self._process_extract_relations(chunks, entities)
    # 识别实体间的关系
    # 示例输出:
    # {
    #   "rel-001": {
    #     "source": "entity-001",
    #     "target": "entity-002",
    #     "type": "uses"
    #   }
    # }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤 3: 存储到知识图谱
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    await self.graph_storage.upsert_nodes(entities)
    await self.graph_storage.upsert_edges(relations)
```

### 知识图谱示例

```
┌─────────────────┐
│   RAG System    │
│  (Technology)   │
└────────┬────────┘
         │ uses
         ▼
┌─────────────────┐
│ Knowledge Graph │
│   (Concept)     │
└─────────────────┘
```

---

## T9: 文档处理完成

```python
# 最终状态更新
await self.doc_status.upsert({
    doc_id: {
        "status": "COMPLETED",
        "chunks_count": 6,
        "entities_count": 15,
        "relations_count": 20,
        "completed_at": datetime.now()
    }
})
```

### 最终存储状态

**Doc Status**:
```json
{
  "doc-xyz789": {
    "status": "COMPLETED",
    "chunks_count": 6,
    "entities_count": 15,
    "relations_count": 20,
    "file_path": "document.pdf",
    "created_at": "2025-10-03T12:00:00",
    "completed_at": "2025-10-03T12:05:00"
  }
}
```

---

## 完整数据流时间线总结

| 时间点 | 阶段 | 负责组件 | 数据形态 | 关键操作 |
|-------|------|---------|---------|----------|
| T0 | 上传 | 用户 | 原始文件 | 提供文档 |
| T1 | 解析 | RAG-Anything | 结构化 JSON | MinerU 解析 |
| T2 | 分离 | RAG-Anything | 完整文本 + 多模态 | 文本合并 |
| T3 | 插入 | RAG-Anything | 完整文本字符串 | 调用 LightRAG API |
| T4 | 入队 | LightRAG | 完整文本（KV存储） | 保存待处理文档 |
| T5 | 启动 | LightRAG | 队列任务 | 后台处理启动 |
| T6 | **✅ 切片** | **LightRAG** | **切片列表** | **滑动窗口算法** |
| T7 | 存储 | LightRAG | 切片 + 向量 | 多层存储 |
| T8 | 提取 | LightRAG | 实体 + 关系 | 知识图谱构建 |
| T9 | 完成 | LightRAG | 完整索引 | 可检索状态 |

---

## 关键结论

### 1. 切片由 LightRAG 完成

**RAG-Anything 的职责**:
- ✅ 文档解析（MinerU）
- ✅ 内容分离（文本 vs 多模态）
- ✅ 调用 LightRAG API
- ❌ **不负责文本切片**

**LightRAG 的职责**:
- ✅ 接收完整文本
- ✅ **执行切片算法**
- ✅ 向量化存储
- ✅ 知识图谱提取

### 2. 切片算法

- **算法**: 滑动窗口（Token-based）
- **窗口大小**: 可配置（默认 1024 tokens）
- **重叠大小**: 可配置（默认 128 tokens）
- **优点**: 保留上下文连续性

### 3. 数据流向

```
RAG-Anything                    LightRAG
─────────────                   ─────────
    解析                            │
     ↓                              │
  完整文本 ──────────────────────→  │
                                   │
                                接收入队
                                   │
                                   ▼
                              ✅ 切片处理
                                   │
                                   ▼
                                向量化
                                   │
                                   ▼
                              知识图谱提取
```

### 4. 配置参数

切片行为可通过以下参数控制：

```python
# lightrag/__init__.py
rag = LightRAG(
    chunking_func=chunking_by_token_size,  # 切片函数
    max_token_size=1024,                    # 最大 token 数
    overlap_token_size=128                  # 重叠 token 数
)
```

---

## 相关文件位置

### RAG-Anything
- 解析器: `raganything/parser.py`
- 内容分离: `raganything/utils.py::separate_content()`
- API 服务: `raganything/api/raganything_server.py::insert_text_content()`

### LightRAG
- 文档处理: `lightrag/lightrag.py::process_document()`
- 切片算法: `lightrag/operate.py::chunking_by_token_size()`
- 管道处理: `lightrag/kg/shared_storage.py::pipeline_process_enqueue_documents()`
- 实体提取: `lightrag/lightrag.py::_process_extract_entities()`

---

## 附录：切片计算公式

### 切片数量计算

给定：
- `total_tokens`: 总 token 数
- `max_token_size`: 切片大小
- `overlap_token_size`: 重叠大小

则：
- `stride = max_token_size - overlap_token_size`
- `num_chunks = ⌈total_tokens / stride⌉`

### 示例计算

```python
total_tokens = 5000
max_token_size = 1024
overlap_token_size = 128

stride = 1024 - 128 = 896
num_chunks = ceil(5000 / 896) = ceil(5.58) = 6
```

### Token 范围计算

```python
for i in range(num_chunks):
    start = i * stride
    end = min(start + max_token_size, total_tokens)
    print(f"Chunk {i}: tokens[{start}:{end}]")
```

输出：
```
Chunk 0: tokens[0:1024]
Chunk 1: tokens[896:1920]
Chunk 2: tokens[1792:2816]
Chunk 3: tokens[2688:3712]
Chunk 4: tokens[3584:4608]
Chunk 5: tokens[4480:5000]
```

---

**文档版本**: v1.0
**最后更新**: 2025-10-03
**维护者**: RAG Core Integration Team
