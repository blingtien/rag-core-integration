# 修复 'hashing_kv' 错误

## 问题描述

在使用 RAG-Anything 处理多模态内容（表格、图片、公式）时，会出现以下错误：

```
ERROR: Error generating table description: 'hashing_kv'
```

## 错误根源

**问题代码位置**: `lightrag/llm/openai.py:499`

```python
async def openai_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
):
    # ...
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    # ❌ KeyError: 'hashing_kv' - 缺少这个参数！
```

**问题原因**:
- LightRAG 的 LLM 函数 (`openai_complete`) **期望** 在 `kwargs` 中有 `hashing_kv` 参数
- RAG-Anything 在调用 LLM 函数时 **没有传递** 这个参数
- 导致 KeyError 异常

## 修复方案

### 方案 1: 修改 RAG-Anything（推荐）✅

在 RAG-Anything 调用 LLM 函数时传递 `hashing_kv` 参数。

#### 修改位置 1: `raganything/modalprocessors.py`

在所有 `self.modal_caption_func()` 调用中添加 `hashing_kv` 参数：

```python
# 当前代码（有问题）
response = await self.modal_caption_func(
    table_prompt,
    system_prompt=PROMPTS["TABLE_ANALYSIS_SYSTEM"],
)

# 修复后代码
response = await self.modal_caption_func(
    table_prompt,
    system_prompt=PROMPTS["TABLE_ANALYSIS_SYSTEM"],
    hashing_kv=self.lightrag.llm_response_cache,  # ✅ 添加 hashing_kv
)
```

**需要修改的函数**:
1. `ImageModalProcessor._generate_image_description()` - 约 3 处
2. `TableModalProcessor._generate_table_description()` - 约 3 处
3. `EquationModalProcessor._generate_equation_description()` - 约 3 处
4. `GenericModalProcessor._generate_generic_description()` - 约 3 处

#### 完整修改示例

```python
# raganything/modalprocessors.py

class TableModalProcessor(BaseModalProcessor):
    async def _generate_table_description(
        self,
        modal_content: str,
        entity_name: str = None,
        table_img_path: str = None,
        table_caption: str = None,
        table_footnote: str = None,
        context: str = None,
    ) -> tuple[str, dict]:
        try:
            # Build table analysis prompt
            if context:
                table_prompt = PROMPTS.get(
                    "table_prompt_with_context", PROMPTS["table_prompt"]
                ).format(
                    context=context,
                    entity_name=entity_name if entity_name else "descriptive name for this table",
                    table_img_path=table_img_path,
                    table_caption=table_caption if table_caption else "None",
                    table_body=table_body,
                    table_footnote=table_footnote if table_footnote else "None",
                )
            else:
                table_prompt = PROMPTS["table_prompt"].format(
                    entity_name=entity_name if entity_name else "descriptive name for this table",
                    table_img_path=table_img_path,
                    table_caption=table_caption if table_caption else "None",
                    table_body=table_body,
                    table_footnote=table_footnote if table_footnote else "None",
                )

            # ✅ 修复：添加 hashing_kv 参数
            response = await self.modal_caption_func(
                table_prompt,
                system_prompt=PROMPTS["TABLE_ANALYSIS_SYSTEM"],
                hashing_kv=self.lightrag.llm_response_cache,  # 新增
            )

            # Parse response
            enhanced_caption, entity_info = self._parse_table_response(
                response, entity_name
            )

            return enhanced_caption, entity_info

        except Exception as e:
            logger.error(f"Error generating table description: {e}")
            # Fallback处理
            fallback_entity = {
                "entity_name": entity_name if entity_name else f"table_{compute_mdhash_id(str(modal_content))}",
                "entity_type": "table",
                "summary": f"Table content: {str(modal_content)[:100]}",
            }
            return str(modal_content), fallback_entity
```

#### 修改步骤

1. **在所有 Modal Processor 中查找 `self.modal_caption_func` 调用**
2. **添加 `hashing_kv` 参数**：
   ```python
   hashing_kv=self.lightrag.llm_response_cache
   ```
3. **测试验证**

### 方案 2: 修改 LightRAG LLM 函数（不推荐）

修改 `lightrag/llm/openai.py` 使 `hashing_kv` 参数可选：

```python
# lightrag/llm/openai.py

async def openai_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    if history_messages is None:
        history_messages = []
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = "json"

    # ✅ 修复：安全获取 hashing_kv
    hashing_kv = kwargs.get("hashing_kv")
    if hashing_kv and hasattr(hashing_kv, 'global_config'):
        model_name = hashing_kv.global_config.get("llm_model_name")
    else:
        # 使用默认模型或从环境变量获取
        model_name = os.environ.get("LLM_MODEL", "gpt-4")

    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
```

**不推荐原因**:
- 需要修改 LightRAG 核心代码
- 可能影响其他使用 LightRAG 的代码
- 不符合 LightRAG 的设计意图

## 实施建议

### 推荐方案：方案 1

**优点**:
- ✅ 不修改 LightRAG 核心代码
- ✅ 遵循 LightRAG 的设计规范
- ✅ 提供正确的缓存支持
- ✅ 修改范围小，风险低

**实施步骤**:

1. **备份当前代码**
   ```bash
   cp raganything/modalprocessors.py raganything/modalprocessors.py.backup
   ```

2. **批量修改 modal_caption_func 调用**

   在 `raganything/modalprocessors.py` 中搜索所有：
   ```python
   await self.modal_caption_func(
   ```

   在参数列表末尾添加：
   ```python
   hashing_kv=self.lightrag.llm_response_cache,
   ```

3. **具体修改位置**（约 12 处）:
   - `ImageModalProcessor._generate_image_description()`: 3 处
   - `TableModalProcessor._generate_table_description()`: 3 处
   - `EquationModalProcessor._generate_equation_description()`: 3 处
   - `GenericModalProcessor._generate_generic_description()`: 3 处

4. **测试验证**
   ```bash
   # 运行多模态处理测试
   python test_multimodal_processing.py
   ```

## 验证方法

### 测试脚本

```python
import asyncio
from raganything import RAGAnything

async def test_multimodal():
    rag = RAGAnything(
        # 配置参数
    )

    # 测试表格处理
    result = await rag.process_document("test_document.pdf")
    print("✅ 多模态处理成功!")
    print(f"处理的表格数量: {result.table_count}")

asyncio.run(test_multimodal())
```

### 预期结果

**修复前**:
```
ERROR: Error generating table description: 'hashing_kv'
```

**修复后**:
```
INFO: Generated descriptions for 7/7 multimodal items using correct processors
✅ 多模态处理成功!
处理的表格数量: 7
```

## 代码差异

### 修改前
```python
response = await self.modal_caption_func(
    table_prompt,
    system_prompt=PROMPTS["TABLE_ANALYSIS_SYSTEM"],
)
```

### 修改后
```python
response = await self.modal_caption_func(
    table_prompt,
    system_prompt=PROMPTS["TABLE_ANALYSIS_SYSTEM"],
    hashing_kv=self.lightrag.llm_response_cache,  # ✅ 添加此行
)
```

## 相关文件

- **问题文件**: `lightrag/llm/openai.py:499`
- **修改文件**: `raganything/modalprocessors.py`
- **涉及类**:
  - `ImageModalProcessor`
  - `TableModalProcessor`
  - `EquationModalProcessor`
  - `GenericModalProcessor`

## 参考

- [LightRAG LLM 接口文档](../lightrag/llm/Readme.md)
- [RAG-Anything 多模态处理器](../raganything/modalprocessors.py)
- [数据流文档](./data-flow-raganything-to-lightrag.md)

---

**文档版本**: v1.0
**创建时间**: 2025-10-03
**最后更新**: 2025-10-03
