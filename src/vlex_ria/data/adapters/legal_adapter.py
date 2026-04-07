from typing import Dict, Any, List, Optional
from datasets import Dataset

from vlex_ria.core.utils import get_logger

logger = get_logger(__name__)

class LegalDatasetAdapter:
    @staticmethod
    def _extract_legal_entities(text: str) -> str:
        """Trích xuất thực thể luật (LER). Ví dụ: Điều 12, Khoản 3, Luật Doanh nghiệp."""
        if not text: return ""
        import re
        pattern = r"(Điều\s+\d+|Khoản\s+\d+|Luật\s+[A-Z][A-Za-z\s]+)"
        matches = re.findall(pattern, text)
        unique_entities = sorted(list(set(matches)))
        return ", ".join(unique_entities) if unique_entities else ""

    @staticmethod
    def _detect_schema(columns: List[str]) -> str:
        """
        Dự đoán schema dựa trên tên các trường trong dataset.
        """
        if "instruction" in columns and "output" in columns:
            return "alpaca"
        if "prompt" in columns and "chosen" in columns and "rejected" in columns:
            return "rlhf"
        if "messages" in columns: # ChatML
            return "chatml"
        if "question" in columns and "context_list" in columns:
            return "legal_qa"
        if "text" in columns:
            return "text"
        if "content" in columns:
            return "content"
        return "unknown"

    @classmethod
    def adapt_for_pretrain(cls, dataset: Dataset, text_column: Optional[str] = None) -> Dataset:
        """
        Chuyển dataset thành định dạng Pretrain (chỉ lấy text thô).
        """
        columns = dataset.column_names
        
        if text_column and text_column in columns:
            pass # Sẽ dùng text_column
        elif "text" in columns:
            text_column = "text"
        elif "output" in columns:
            text_column = "output"
        elif "content" in columns:
            text_column = "content"
        elif "question" in columns and "context_list" in columns:
            # Legal QA: gộp question + contexts thành text 
            logger.info("Legal QA schema detected for pretrain: merging question + context_list")
            def _merge_legal_qa(example):
                question = example.get("question", "")
                contexts = example.get("context_list", [])
                if isinstance(contexts, list):
                    ctx_text = "\n\n".join(str(c) for c in contexts if c)
                else:
                    ctx_text = str(contexts) if contexts else ""
                return {"text": f"{question}\n\n{ctx_text}".strip()}
            return dataset.map(_merge_legal_qa, remove_columns=columns)
        else:
            logger.warning(f"Không tự động tìm thấy cột text cho pretrain. Thử gộp tất cả cột string.")
            def _merge_cols(example):
                merged = []
                for k, v in example.items():
                    if isinstance(v, str) and v.strip() != "":
                        merged.append(f"{k}: {v}")
                return {"text": "\n\n".join(merged)}
            return dataset.map(_merge_cols, remove_columns=columns)
            
        def _to_text(example):
            val = example[text_column]
            if isinstance(val, list):
                val = "\n\n".join(str(v) for v in val if v)
            return {"text": str(val) if val else ""}
            
        return dataset.map(_to_text, remove_columns=[c for c in columns if c != "text"])

    @classmethod
    def adapt_for_sft(cls, dataset: Dataset) -> Dataset:
        """
        Chuyển sang định dạng SFT (instruction, input, output).
        """
        columns = dataset.column_names
        schema = cls._detect_schema(columns)
        
        if schema == "alpaca":
            return dataset

        logger.info(f"Đang convert schema {schema} sang SFT (alpaca).")
        
        def _map_to_alpaca(example: Dict[str, Any]) -> Dict[str, Any]:
            res = {"instruction": "", "input": "", "output": ""}
            
            if schema == "legal_qa":
                # YuITC/Vietnamese-Legal-Documents: question + context_list
                question = example.get("question", "")
                contexts = example.get("context_list", [])
                if isinstance(contexts, list):
                    ctx_text = "\n\n".join(str(c) for c in contexts[:3] if c)  # Take top 3 contexts
                else:
                    ctx_text = str(contexts) if contexts else ""
                
                # Phase 3: Legal Entity Recognition
                entities = cls._extract_legal_entities(ctx_text)
                if entities:
                    ctx_text = f"[Thực thể pháp lý liên quan: {entities}]\n\n{ctx_text}"
                
                res["instruction"] = question
                res["input"] = ctx_text
                res["output"] = f"Dựa trên các quy định pháp luật được cung cấp, trường hợp của bạn được quy định như sau:\n\n{ctx_text}" if ctx_text else "Không tìm thấy thông tin liên quan."
            
            elif schema == "rlhf":
                prompt = example.get("prompt", "")
                chosen = example.get("chosen", "")
                
                # Trích xuất instruction từ "Human: ...\n\nAssistant:" nêú có
                if "Human:" in chosen and "Assistant:" in chosen:
                    parts = chosen.split("Assistant:")
                    human_part = parts[0].replace("Human:", "").strip()
                    assistant_part = parts[1].strip()
                    res["instruction"] = human_part
                    res["output"] = assistant_part
                else:
                    res["instruction"] = prompt
                    res["output"] = chosen
                    
            elif schema == "chatml":
                msgs = example.get("messages", [])
                instr = ""
                out = ""
                for m in msgs:
                    if m.get("role") in ["user", "human"]:
                        instr += m.get("content", "") + "\n"
                    elif m.get("role") in ["assistant", "bot"]:
                        out += m.get("content", "") + "\n"
                res["instruction"] = instr.strip()
                res["output"] = out.strip()
                
            else:
                # Fallback ngẫu nhiên từ những text fields
                texts = [str(v) for k, v in example.items() if isinstance(v, str)]
                if len(texts) >= 2:
                    res["instruction"] = texts[0]
                    res["output"] = texts[1]
                elif len(texts) == 1:
                    res["instruction"] = texts[0]
                    res["output"] = "N/A"
            return res

        return dataset.map(_map_to_alpaca, remove_columns=columns)

    @classmethod
    def adapt_for_rl(cls, dataset: Dataset) -> Dataset:
        """
        Chuyển sang định dạng RL (HH-RLHF format hoặc lấy field raw prompt).
        Thực ra RLDataset của VLexRIA chờ đợi cột 'chosen' có "Human:... Assistant:..."
        """
        columns = dataset.column_names
        schema = cls._detect_schema(columns)
        
        def _map_to_rl(example: Dict[str, Any]) -> Dict[str, Any]:
            res = {"chosen": ""}
            
            if schema == "legal_qa":
                # YuITC/Vietnamese-Legal-Documents: question + context_list
                question = example.get("question", "")
                contexts = example.get("context_list", [])
                if isinstance(contexts, list):
                    ctx_text = "\n\n".join(str(c) for c in contexts[:3] if c)
                else:
                    ctx_text = str(contexts) if contexts else ""
                user_msg = question
                if ctx_text:
                    user_msg += f"\n\nNgữ cảnh pháp lý:\n{ctx_text}"
                
                output_msg = f"Dựa trên các quy định pháp luật hiện hành, trường hợp của bạn được quy định như sau:\n\n{ctx_text}" if ctx_text else "Hiện tại không tìm thấy căn cứ pháp lý."
                
                res["chosen"] = f"Human: {user_msg}\n\nAssistant: {output_msg}"
                res["rejected"] = f"Human: {user_msg}\n\nAssistant: Tôi không biết câu trả lời cho vấn đề pháp lý này."
            
            elif schema == "rlhf":
                res["chosen"] = example.get("chosen", "")
            elif schema == "alpaca":
                instr = example.get("instruction", "")
                inp = example.get("input", "")
                out = example.get("output", "")
                
                user_msg = instr
                if inp:
                    user_msg += f"\n{inp}"
                
                res["chosen"] = f"Human: {user_msg}\n\nAssistant: {out}"
            else:
                texts = [v for k, v in example.items() if isinstance(v, str)]
                if len(texts) >= 2:
                    res["chosen"] = f"Human: {texts[0]}\n\nAssistant: {texts[1]}"
                elif len(texts) == 1:
                    res["chosen"] = f"Human: \n\nAssistant: {texts[0]}"
                    
            return res

        return dataset.map(_map_to_rl, remove_columns=columns)
