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
        "请{}",
        "能不能{}",
        "能帮我{}吗",
        "我想{}",
        "麻烦{}一下",
        "现在需要{}"
    ],
    
    "START_RECORDING": [
        "{}",
        "请{}",
        "帮我{}",
        "开始{}",
        "现在{}",
        "我想{}",
        "能否{}",
        "麻烦{}"
    ],
    
    "STOP_RECORDING": [
        "{}",
        "请{}",
        "现在{}",
        "帮我{}",
        "可以{}了",
        "我想{}",
        "{}"
    ],
    
    "CAPTURE_AND_DESCRIBE": [
        "{}",
        "请{}",
        "能否{}",
        "帮我{}",
        "我想知道{}",
        "麻烦{}",
        "请问{}"
    ],
    
    "CAPTURE_AND_REMEMBER": [
        "{}",
        "请{}",
        "帮我{}",
        "希望{}",
        "需要{}",
        "想要{}"
    ],
    
    "CAPTURE_SCAN_QR": [
        "{}",
        "请{}",
        "帮我{}",
        "能不能{}",
        "麻烦{}"
    ],
    
    "GET_BATTERY_LEVEL": [
        "{}",
        "请问{}",
        "能告诉我{}吗",
        "查询{}",
        "我想知道{}"
    ],
    
    "OTHERS": [
        "{}",
        "请{}",
        "帮我{}"
    ],
    
    "DEFAULT": [
        "{}",
        "请{}",
        "帮我{}",
        "能否{}",
        "可以{}吗",
        "我想{}"
    ]
}

# 终止词列表
TERMINATORS = ["", "。", "吧", "了", "哦", "呢", "啊", "呀"]

def apply_template(phrase, templates=None):
    """
    应用模板生成更多样化的提示语
    
    参数:
        phrase: 基础短语
        templates: 模板列表，如果为None则使用默认模板
    
    返回:
        应用模板后的新提示语
    """
    if templates is None:
        templates = TEMPLATES["DEFAULT"]
    
    template = random.choice(templates)
    return template.format(phrase)

def add_terminator(phrase):
    """
    添加终止词
    
    参数:
        phrase: 语句
    
    返回:
        添加终止词后的语句
    """
    return phrase + random.choice(TERMINATORS)

def generate_prompt_variants(base_prompts, intent, num_variants=5):
    """
    为基础提示语生成变体
    
    参数:
        base_prompts: 基础提示语列表
        intent: 意图类别
        num_variants: 每个基础提示语生成的变体数量
        
    返回:
        生成的提示语变体列表
    """
    variants = []
    templates = TEMPLATES.get(intent, TEMPLATES["DEFAULT"])
    
    for base_prompt in base_prompts:
        # 添加原始提示语
        variants.append(base_prompt)
        
        # 生成变体
        for _ in range(num_variants):
            # 应用模板
            variant = apply_template(base_prompt, templates)
            
            # 添加终止词
            variant = add_terminator(variant)
            
            # 确保不重复添加
            if variant not in variants and variant not in base_prompts:
                variants.append(variant)
    
    return variants

def enrich_intent_prompts(output_file="expanded_prompts.json", variants_per_prompt=3):
    """
    扩充所有意图的提示语并保存到文件
    
    参数:
        output_file: 输出文件路径
        variants_per_prompt: 每个基础提示语生成的变体数量
    """
    enriched_prompts = {}
    
    for intent in INTENT_CLASSES:
        base_prompts = get_all_prompts_for_intent(intent)
        variants = generate_prompt_variants(base_prompts, intent, variants_per_prompt)
        enriched_prompts[intent] = variants
        
        print(f"{intent}: 从{len(base_prompts)}个基础提示语扩展到{len(variants)}个变体")
    
    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enriched_prompts, f, ensure_ascii=False, indent=2)
    
    print(f"扩充的提示语已保存到: {output_file}")
    return enriched_prompts

def update_intent_prompts_module(enriched_prompts, output_file="new_intent_prompts.py"):
    """
    更新intent_prompts.py模块
    
    参数:
        enriched_prompts: 扩充后的提示语字典
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('"""\n意图和提示语列表\n提供8种意图类别的常用口语表达方式\n"""\n\n')
        f.write('import random\n\n')
        
        # 写入意图类别
        f.write('# 意图类别\n')
        f.write('INTENT_CLASSES = [\n')
        for intent in INTENT_CLASSES:
            f.write(f'    "{intent}",\n')
        f.write(']\n\n')
        
        # 写入提示语字典
        f.write('# 每个意图类别的提示语列表\n')
        f.write('INTENT_PROMPTS = {\n')
        for intent, prompts in enriched_prompts.items():
            f.write(f'    "{intent}": [\n')
            for prompt in prompts:
                f.write(f'        "{prompt}",\n')
            f.write('    ],\n    \n')
        f.write('}\n\n')
        
        # 写入辅助函数
        f.write('def get_random_prompt(intent):\n')
        f.write('    """获取指定意图的随机提示语"""\n')
        f.write('    if intent in INTENT_PROMPTS:\n')
        f.write('        return random.choice(INTENT_PROMPTS[intent])\n')
        f.write('    return "未知意图"\n\n')
        
        f.write('def get_all_prompts_for_intent(intent):\n')
        f.write('    """获取指定意图的所有提示语"""\n')
        f.write('    if intent in INTENT_PROMPTS:\n')
        f.write('        return INTENT_PROMPTS[intent]\n')
        f.write('    return []\n')
    
    print(f"已更新提示语模块: {output_file}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='EdgeVoice提示语生成器')
    parser.add_argument('--output_json', type=str, default='expanded_prompts.json',
                        help='输出JSON文件路径')
    parser.add_argument('--output_py', type=str, default='new_intent_prompts.py',
                        help='输出Python模块文件路径')
    parser.add_argument('--variants', type=int, default=3,
                        help='每个基础提示语生成的变体数量')
    
    args = parser.parse_args()
    
    # 扩充提示语
    enriched_prompts = enrich_intent_prompts(args.output_json, args.variants)
    
    # 更新Python模块
    update_intent_prompts_module(enriched_prompts, args.output_py)

if __name__ == "__main__":
    main() 