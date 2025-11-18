import os
import sys
import json
import re
from typing import List, Dict, Any, Optional

project_root_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root_path not in sys.path:
    sys.path.append(project_root_path)
if os.path.dirname(project_root_path) not in sys.path:
    sys.path.append(os.path.dirname(project_root_path))

from agent.llms import AbstractLLM


class TPCLLM(AbstractLLM):
    """
    Travel Planning Challenge LLM
    
    使用本地部署的 Qwen3-4B-Instruct 模型
    """
    
    def __init__(self, model_name: str = "qwen", model_path: Optional[str] = None,
                 device: str = "cuda", **kwargs):
        """
        初始化 TPC LLM
        
        Args:
            model_name: 模型名称
            model_path: 本地模型路径 (默认: D:\\models\\Qwen3-4B-Instruct)
            device: 使用设备 ('cuda' 强制使用GPU)
            **kwargs: 其他模型配置
        """
        super().__init__()
        self.name = "TPCLLM"
        self.model_name = model_name
        
        # 设置本地模型路径
        if model_path is None:
            # 默认使用您的本地模型路径
            self.model_path = r"D:\models\Qwen3-4B-Instruct"
        else:
            self.model_path = model_path
            
        self.device = device
        self.config = kwargs
        
        # 模型和分词器
        self.model = None
        self.tokenizer = None
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载本地语言模型"""
        
        print(f"[TPCLLM] 正在从本地加载模型...")
        print(f"[TPCLLM] 模型路径: {self.model_path}")
        print(f"[TPCLLM] 使用设备: {self.device}")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # 检查 CUDA 是否可用
            if not torch.cuda.is_available():
                print("[警告] CUDA 不可用！请检查:")
                print("  1. 是否安装了 CUDA 版本的 PyTorch")
                print("  2. NVIDIA 驱动是否正确安装")
                print("  3. 使用以下命令检查: python -c 'import torch; print(torch.cuda.is_available())'")
                raise RuntimeError("CUDA not available")
            
            print(f"[TPCLLM] ✓ 检测到 {torch.cuda.device_count()} 个 GPU")
            print(f"[TPCLLM] ✓ GPU 型号: {torch.cuda.get_device_name(0)}")
            
            # 加载分词器
            print("[TPCLLM] 正在加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            print("[TPCLLM] ✓ 分词器加载成功")
            
            # 加载模型到 GPU
            print("[TPCLLM] 正在加载模型到 GPU (这可能需要几分钟)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="cuda:0",  # 强制使用第一个GPU
                trust_remote_code=True,
                torch_dtype=torch.float16,  # 使用半精度加速
                # 对于 4B 模型，通常不需要量化，但如果显存不够可以启用：
                load_in_8bit=True,
            )
            
            print(f"[TPCLLM] ✓ 模型加载成功！")
            print(f"[TPCLLM] ✓ 模型已加载到: {self.model.device}")
            
            # 显示显存使用情况
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                print(f"[TPCLLM] GPU 显存使用: {allocated:.2f}GB / {reserved:.2f}GB")
            
        except FileNotFoundError:
            print(f"[错误] 找不到模型文件！")
            print(f"请检查路径是否正确: {self.model_path}")
            print("当前目录应包含以下文件:")
            print("  - config.json")
            print("  - tokenizer.json")
            print("  - model-*.safetensors")
            self.model = None
            self.tokenizer = None
            
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"[错误] GPU 相关错误: {e}")
                print("\n解决方案:")
                print("1. 确认安装了 CUDA 版本的 PyTorch:")
                print("   pip uninstall torch")
                print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                print("2. 检查 NVIDIA 驱动是否正确安装")
                print("3. 重启电脑后再试")
            else:
                print(f"[错误] 模型加载失败: {e}")
            self.model = None
            self.tokenizer = None
            
        except Exception as e:
            print(f"[错误] 模型加载失败: {e}")
            print("\n可能的原因:")
            print("1. 模型文件不完整或损坏")
            print("2. 显存不足（Qwen3-4B 需要约 8-10GB 显存）")
            print("3. PyTorch 或 transformers 版本不兼容")
            print("\n[警告] 将使用回退模式（基于规则的响应）")
            self.model = None
            self.tokenizer = None
    
    def _get_response(self, messages, one_line=False, json_mode=False):
        """
        生成模型响应
        
        Args:
            messages: 输入提示（str）或聊天消息列表（list）
            one_line: 如果为 True，只返回第一行
            json_mode: 如果为 True，解析响应为 JSON
            
        Returns:
            生成的响应字符串
        """
        
        # 将消息转换为提示格式
        if isinstance(messages, list):
            prompt = messages  # 保持列表格式用于 chat template
        else:
            prompt = [{"role": "user", "content": str(messages)}]
        
        # 生成响应
        if self.model is not None and self.tokenizer is not None:
            # 使用实际模型推理
            response = self._model_inference(prompt, json_mode)
        else:
            # 使用回退生成（基于规则）
            prompt_text = self._format_chat_messages(prompt) if isinstance(prompt, list) else prompt
            response = self._fallback_generation(prompt_text, json_mode)
        
        # 后处理
        if json_mode:
            response = self._extract_json(response)
        
        if one_line:
            response = response.split("\n")[0]
        
        return response
    
    def _format_chat_messages(self, messages: List[Dict]) -> str:
        """将聊天消息格式化为单个提示"""
        prompt_parts = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts)
    
    def _model_inference(self, messages: List[Dict], json_mode: bool = False) -> str:
        """
        执行实际的模型推理（使用 GPU）
        
        Args:
            messages: 聊天消息列表
            json_mode: 是否需要 JSON 格式输出
            
        Returns:
            生成的响应
        """
        import torch
        
        try:
            # 如果需要 JSON 输出，添加系统提示
            if json_mode:
                system_msg = {
                    "role": "system",
                    "content": "你必须返回有效的JSON格式，不要添加其他文字说明。只输出JSON，不要有任何额外的解释。"
                }
                # 检查是否已有系统消息
                if not messages or messages[0].get('role') != 'system':
                    messages = [system_msg] + messages
            
            # 应用聊天模板
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 分词并移到 GPU
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            # 生成（在 GPU 上）
            with torch.no_grad():
                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=2048,  # 最大生成长度
                    temperature=0.7,      # 控制随机性
                    top_p=0.9,           # nucleus sampling
                    do_sample=True,      # 启用采样
                    pad_token_id=self.tokenizer.eos_token_id,
                    # 可选：添加更多控制
                    # repetition_penalty=1.1,  # 避免重复
                    # num_beams=1,  # beam search
                )
            
            # 解码（只取新生成的部分）
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return response
            
        except torch.cuda.OutOfMemoryError:
            print("[错误] GPU 显存不足！")
            print("建议:")
            print("1. 减少 max_new_tokens (当前: 2048)")
            print("2. 启用 8-bit 量化: load_in_8bit=True")
            print("3. 关闭其他占用显存的程序")
            # 清理显存
            torch.cuda.empty_cache()
            # 回退到规则生成
            prompt_text = self._format_chat_messages(messages)
            return self._fallback_generation(prompt_text, json_mode)
            
        except Exception as e:
            print(f"[错误] 模型推理失败: {e}")
            # 回退到规则生成
            prompt_text = self._format_chat_messages(messages)
            return self._fallback_generation(prompt_text, json_mode)
    
    def _fallback_generation(self, prompt: str, json_mode: bool = False) -> str:
        """
        回退生成（使用基于规则的方法）
        
        当模型加载失败时使用。
        """
        
        # 从提示中检测意图
        prompt_lower = prompt.lower()
        
        # 约束提取
        if "提取" in prompt or "extract" in prompt_lower:
            return self._generate_constraint_json(prompt)
        
        # 每日计划生成
        elif "第" in prompt and "天" in prompt and "计划" in prompt:
            return self._generate_day_plan_json(prompt)
        
        # 默认 JSON 响应
        elif json_mode:
            return '{"message": "回退响应。模型未加载。", "status": "fallback"}'
        
        # 默认文本响应
        else:
            return "这是一个回退响应。模型未成功加载到 GPU。"
    
    def _generate_constraint_json(self, prompt: str) -> str:
        """生成约束提取 JSON（回退）"""
        
        constraints = {
            "destination": None,
            "duration_days": 3,
            "budget": None,
            "must_visit": [],
            "avoid": [],
            "min_attractions_per_day": 3,
            "preferences": []
        }
        
        # 尝试提取目的地
        cities = ["北京", "上海", "杭州", "西安", "成都", "广州", "深圳", "南京", "重庆", "武汉"]
        for city in cities:
            if city in prompt:
                constraints["destination"] = city
                break
        
        # 提取时长
        day_match = re.search(r'(\d+)[天日]', prompt)
        if day_match:
            constraints["duration_days"] = int(day_match.group(1))
        
        # 提取预算
        budget_match = re.search(r'预算?\s*[:：]?\s*(\d+)', prompt)
        if budget_match:
            constraints["budget"] = int(budget_match.group(1))
        
        # 提取必访景点
        for place in ["故宫", "长城", "天坛", "外滩", "东方明珠", "西湖", "灵隐寺"]:
            if place in prompt:
                constraints["must_visit"].append(place)
        
        # 提取最少景点数
        attr_match = re.search(r'至少\s*(\d+)\s*个?景点', prompt)
        if attr_match:
            constraints["min_attractions_per_day"] = int(attr_match.group(1))
        
        return json.dumps(constraints, ensure_ascii=False, indent=2)
    
    def _generate_day_plan_json(self, prompt: str) -> str:
        """生成每日计划 JSON（回退）"""
        
        # 提取天数
        day_match = re.search(r'第(\d+)天', prompt)
        day = int(day_match.group(1)) if day_match else 1
        
        # 提取目的地
        cities = ["北京", "上海", "杭州", "西安", "成都"]
        destination = "北京"
        for city in cities:
            if city in prompt:
                destination = city
                break
        
        # 提取日期
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', prompt)
        date = date_match.group(1) if date_match else "2025-11-01"
        
        # 生成计划
        day_plan = {
            "day": day,
            "date": date,
            "current_city": destination,
            "transportation": None,
            "breakfast": f"{destination}特色早餐店",
            "attraction": [
                f"{destination}景点A",
                f"{destination}景点B",
                f"{destination}景点C"
            ],
            "lunch": f"{destination}特色餐厅",
            "dinner": f"{destination}美食广场",
            "accommodation": f"{destination}酒店"
        }
        
        return json.dumps(day_plan, ensure_ascii=False, indent=2)
    
    def _extract_json(self, response: str) -> str:
        """从响应文本中提取 JSON"""
        
        # 尝试查找 JSON 对象
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = list(re.finditer(json_pattern, response, re.DOTALL))
        
        for match in matches:
            json_str = match.group(0)
            try:
                # 验证 JSON
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                continue
        
        # 尝试查找 JSON 数组
        array_pattern = r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'
        matches = list(re.finditer(array_pattern, response, re.DOTALL))
        
        for match in matches:
            json_str = match.group(0)
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                continue
        
        # 如果没找到有效的 JSON，返回整个响应
        return response.strip()
    
    def get_response(self, prompt, one_line=False, json_mode=False):
        """
        获取响应的公共接口
        （由 agent 使用）
        """
        return self._get_response(prompt, one_line, json_mode)


# 测试代码
if __name__ == "__main__":
    print("="*70)
    print("测试 TPCLLM - 本地 Qwen3-4B-Instruct 模型")
    print("="*70)
    print()
    
    # 创建 LLM 实例
    print("正在初始化模型...")
    llm = TPCLLM()
    print()
    
    if llm.model is None:
        print("⚠️  模型加载失败，将使用回退模式进行测试")
        print()
    else:
        print("✓ 模型加载成功！开始测试...")
        print()
    
    # 测试 1：约束提取
    print("="*70)
    print("测试 1: 约束提取")
    print("="*70)
    
    query = "我想去北京玩3天,预算3000元,想去故宫和长城,每天至少要去4个景点"
    prompt = f"从用户查询中提取约束: {query}"
    
    print(f"查询: {query}")
    print(f"\n正在生成响应...")
    response = llm.get_response(prompt, json_mode=True)
    print(f"\n响应:")
    print(response)
    
    # 测试 2：每日计划生成
    print("\n" + "="*70)
    print("测试 2: 每日计划生成")
    print("="*70)
    
    prompt = "为第1天生成旅行计划，目的地：北京，日期：2025-11-01"
    print(f"提示: {prompt}")
    print(f"\n正在生成响应...")
    response = llm.get_response(prompt, json_mode=True)
    print(f"\n响应:")
    print(response)
    
    # 测试 3：自由对话
    print("\n" + "="*70)
    print("测试 3: 自由对话")
    print("="*70)
    
    prompt = "推荐一下北京的美食"
    print(f"提示: {prompt}")
    print(f"\n正在生成响应...")
    response = llm.get_response(prompt)
    print(f"\n响应:")
    print(response)
    
    print("\n" + "="*70)
    print("测试完成!")
    print("="*70)
    
    if llm.model is not None:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"\n最终 GPU 显存使用: {allocated:.2f}GB")