"""
提示语生成器
用于生成更多样化的语音提示语
"""

import random
import json
import os
from pathlib import Path
from intent_prompts import INTENT_CLASSES, INTENT_PROMPTS, get_all_prompts_for_intent

# 常用句式模板
TEMPLATES = {
    "TAKE_PHOTO": [
        "{}",
        "帮我{}",
        "能不能{}",
        "嘿，{}一下",
        "眼镜，{}",
        "智能眼镜，请{}",
        "麻烦{}一下",
        "我想要{}"
    ],
    
    "START_RECORDING": [
        "{}",
        "帮我{}",
        "能不能{}",
        "嘿，{}一下",
        "眼镜，{}",
        "智能眼镜，请{}",
        "麻烦{}一下",
        "我想要{}"
    ],
    
    "STOP_RECORDING": [
        "{}",
        "帮我{}",
        "能不能{}",
        "嘿，{}一下",
        "眼镜，{}",
        "智能眼镜，请{}",
        "麻烦{}一下",
        "现在可以{}"
    ],
    
    "CAPTURE_AND_DESCRIBE": [
        "{}",
        "帮我{}",
        "眼镜，能不能{}",
        "智能眼镜，{}",
        "我想要{}",
        "麻烦{}一下",
        "嘿，能{}吗"
    ],
    
    "CAPTURE_AND_REMEMBER": [
        "{}",
        "帮我{}",
        "眼镜，能不能{}",
        "智能眼镜，{}",
        "我想要{}",
        "麻烦{}一下",
        "嘿，能{}吗"
    ],
    
    "CAPTURE_SCAN_QR": [
        "{}",
        "帮我{}",
        "眼镜，能不能{}",
        "智能眼镜，{}",
        "我想要{}",
        "麻烦{}一下",
        "嘿，能{}吗",
        "帮忙{}"
    ],
    
    "GET_BATTERY_LEVEL": [
        "{}",
        "帮我{}",
        "嘿，{}",
        "眼镜，{}",
        "智能眼镜，{}",
        "麻烦{}一下",
        "能不能{}"
    ],
    
    "OTHERS": [
        "{}",
        "帮我{}",
        "嘿，{}",
        "眼镜，{}",
        "智能眼镜，{}",
        "麻烦{}一下",
        "能不能{}",
        "请帮我{}"
    ],
    
    "DEFAULT": [
        "{}",
        "帮我{}",
        "嘿，{}",
        "眼镜，{}",
        "智能眼镜，{}",
        "麻烦{}一下",
        "能不能{}",
        "请帮我{}"
    ]
}

# 终止词列表，更自然的口语表达
TERMINATORS = ["", "吧", "一下", "好吗", "好不好", "可以吗", "行吗", "怎么样"]

def apply_template(phrase, templates=None):
    """
    应用模板生成更符合智能眼镜语音交互的提示语
    
    参数:
        phrase: 基础短语
        templates: 模板列表，如果为None则使用默认模板
    
    返回:
        应用模板后的新提示语，更贴近自然口语交流
    """
    if templates is None:
        templates = TEMPLATES["DEFAULT"]
    
    template = random.choice(templates)
    return template.format(phrase)

def add_terminator(phrase):
    """
    添加自然的语音终止词，增强对话的自然度
    
    参数:
        phrase: 原始提示语
    
    返回:
        添加终止词后的提示语
    """
    terminator = random.choice(TERMINATORS)
    if terminator:
        return f"{phrase}{terminator}"
    return phrase

def generate_prompt_variants(base_prompts, intent, num_variants=5):
    """
    基于基础提示语生成更多变体，增加语音识别的多样性
    
    参数:
        base_prompts: 基础提示语列表
        intent: 意图类别
        num_variants: 每个基础提示语生成的变体数量
    
    返回:
        包含原始提示语和变体的扩展列表
    """
    templates = TEMPLATES.get(intent, TEMPLATES["DEFAULT"])
    variants = set(base_prompts)  # 使用集合避免重复
    
    for base_prompt in base_prompts:
        for _ in range(num_variants):
            # 应用模板
            variant = apply_template(base_prompt, templates)
            # 添加终止词
            variant = add_terminator(variant)
            variants.add(variant)
    
    return list(variants)

def enrich_intent_prompts(output_file="expanded_prompts.json", variants_per_prompt=3):
    """
    扩充意图提示语库，生成更多口语化的表达方式
    
    参数:
        output_file: 输出文件路径
        variants_per_prompt: 每个基础提示语生成的变体数量
    
    返回:
        扩充后的提示语字典
    """
    try:
        from tools.intent_prompts import INTENT_PROMPTS, INTENT_CLASSES
    except ImportError:
        try:
            from intent_prompts import INTENT_PROMPTS, INTENT_CLASSES
        except ImportError:
            print("无法导入intent_prompts模块，请确保该文件在tools目录或当前目录下")
            return {}

    enriched_prompts = {}
    
    print("开始扩充意图提示语...")
    
    # 遍历每个意图类别
    for intent in INTENT_CLASSES:
        if intent in INTENT_PROMPTS:
            base_prompts = INTENT_PROMPTS[intent]
            # 生成变体
            variants = generate_prompt_variants(base_prompts, intent, variants_per_prompt)
            enriched_prompts[intent] = variants
            print(f"  '{intent}': 从{len(base_prompts)}个基础提示语扩充到{len(variants)}个变体")
    
    # 保存到文件
    if output_file:
        import json
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(enriched_prompts, f, ensure_ascii=False, indent=2)
        print(f"扩充后的提示语已保存到 {output_file}")
    
    return enriched_prompts

def update_intent_prompts_module(enriched_prompts, output_file="new_intent_prompts.py"):
    """
    更新intent_prompts模块，将扩充后的提示语写入新文件
    
    参数:
        enriched_prompts: 扩充后的提示语字典
        output_file: 输出文件路径
    """
    intent_classes = list(enriched_prompts.keys())
    
    # 生成新的Python模块内容
    content = [
        '"""',
        '智能眼镜语音助手意图和提示语列表',
        '提供多种意图类别的常用口语表达方式',
        '自动生成的增强版提示语库',
        '"""',
        '',
        'import random',
        '',
        '# 意图类别',
        'INTENT_CLASSES = [',
    ]
    
    # 添加意图类别
    for intent in intent_classes:
        content.append(f'    "{intent}",')
    content.append(']')
    content.append('')
    
    # 添加提示语字典
    content.append('# 每个意图类别的提示语列表')
    content.append('INTENT_PROMPTS = {')
    
    for intent, prompts in enriched_prompts.items():
        content.append(f'    "{intent}": [')
        for prompt in prompts:
            content.append(f'        "{prompt}",')
        content.append('    ],')
        content.append('    ')
    
    content.append('}')
    content.append('')
    
    # 添加辅助函数
    content.extend([
        'def get_random_prompt(intent):',
        '    """获取指定意图的随机提示语"""',
        '    if intent in INTENT_PROMPTS:',
        '        return random.choice(INTENT_PROMPTS[intent])',
        '    return "未知意图"',
        '',
        'def get_all_prompts_for_intent(intent):',
        '    """获取指定意图的所有提示语"""',
        '    if intent in INTENT_PROMPTS:',
        '        return INTENT_PROMPTS[intent]',
        '    return []',
    ])
    
    # 写入文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.write('\n'.join(content))
    
    print(f"已更新意图提示语模块，保存到 {output_file}")

def main():
    """
    主函数，生成扩充的提示语并更新模块
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="智能眼镜语音助手提示语生成工具")
    parser.add_argument("--variants", type=int, default=5, help="每个基础提示语生成的变体数量")
    parser.add_argument("--json_output", type=str, default="expanded_prompts.json", help="JSON输出文件路径")
    parser.add_argument("--py_output", type=str, default="new_intent_prompts.py", help="Python模块输出文件路径")
    args = parser.parse_args()
    
    # 扩充提示语
    enriched_prompts = enrich_intent_prompts(args.json_output, args.variants)
    
    # 更新模块
    update_intent_prompts_module(enriched_prompts, args.py_output)
    
    print(f"提示语生成完成! 生成了{sum(len(v) for v in enriched_prompts.values())}个提示语变体")
    print("可以使用以下命令来测试生成的提示语:")
    print(f"  python -c \"from {args.py_output.replace('.py', '')} import get_random_prompt; print(get_random_prompt('TAKE_PHOTO'))\"")

if __name__ == "__main__":
    main() 