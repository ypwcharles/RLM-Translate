#!/usr/bin/env python3
"""
DMXAPI 连接测试脚本

验证 DMXAPI 中转站连接是否正常工作。

Usage:
    python scripts/test_dmxapi.py
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 自动加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass  # python-dotenv 未安装则跳过

from src.core.dmxapi_client import DMXAPIClient, DMXAPIClientManager


def test_single_client():
    """测试单个客户端。"""
    print("=" * 60)
    print("测试 1: 单个 DMXAPIClient")
    print("=" * 60)
    
    try:
        client = DMXAPIClient(model="gemini-2.0-flash")
        print(f"✓ 客户端创建成功")
        print(f"  模型: {client.model}")
        print(f"  Base URL: {client.base_url}")
        
        # 发送测试请求
        print("\n发送测试请求...")
        response = client.generate("请用一句话介绍你自己。")
        
        print(f"\n✓ 请求成功!")
        print(f"  响应: {response.text[:200]}...")
        print(f"  Token 使用: {response.usage}")
        
    except ValueError as e:
        print(f"✗ 配置错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        return False
        
    return True


def test_client_manager():
    """测试客户端管理器。"""
    print("\n" + "=" * 60)
    print("测试 2: DMXAPIClientManager")
    print("=" * 60)
    
    try:
        manager = DMXAPIClientManager()
        print(f"✓ 管理器创建成功")
        
        # 测试各个角色
        roles = ["analyzer", "drafter", "critic", "editor"]
        for role in roles:
            client = getattr(manager, role)
            print(f"  {role}: {client.model}")
            
        # 使用 drafter 发送测试请求
        print("\n使用 drafter 发送测试请求...")
        response = manager.drafter.generate(
            "Translate to Chinese: The quick brown fox jumps over the lazy dog."
        )
        
        print(f"\n✓ 请求成功!")
        print(f"  翻译结果: {response.text}")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False
        
    return True


def test_translation_workflow():
    """测试翻译工作流。"""
    print("\n" + "=" * 60)
    print("测试 3: 简单翻译工作流")
    print("=" * 60)
    
    source_text = """
    The universe is always stranger than you think. That had been the 
    favorite phrase of a professor of Elvi's back in her graduate study days.
    """
    
    try:
        manager = DMXAPIClientManager()
        
        # Step 1: Drafter
        print("\n[1/3] Drafter 初翻...")
        draft_response = manager.drafter.generate(
            f"将以下英文翻译成中文，保持文学风格：\n\n{source_text}"
        )
        print(f"初稿: {draft_response.text[:100]}...")
        
        # Step 2: Critic
        print("\n[2/3] Critic 审查...")
        critic_response = manager.critic.generate(
            f"审查以下翻译是否准确流畅：\n\n原文：{source_text}\n\n译文：{draft_response.text}\n\n如果翻译质量好，回复'翻译质量良好'；否则指出问题。"
        )
        print(f"审查结果: {critic_response.text[:100]}...")
        
        # Step 3: Editor
        print("\n[3/3] Editor 润色...")
        editor_response = manager.editor.generate(
            f"根据审查意见润色译文：\n\n原文：{source_text}\n\n初稿：{draft_response.text}\n\n审查意见：{critic_response.text}\n\n请输出最终润色后的译文。"
        )
        print(f"最终译文: {editor_response.text}")
        
        print("\n✓ 翻译工作流测试完成!")
        
    except Exception as e:
        print(f"✗ 工作流测试失败: {e}")
        return False
        
    return True


def main():
    """主函数。"""
    print("DMXAPI 连接测试")
    print("=" * 60)
    
    # 检查环境变量
    api_key = os.environ.get("DMXAPI_KEY")
    if not api_key:
        print("⚠️  未设置 DMXAPI_KEY 环境变量")
        print("请先设置: export DMXAPI_KEY='sk-xxx'")
        sys.exit(1)
        
    print(f"API Key: {api_key[:10]}...{api_key[-4:]}")
    print()
    
    # 运行测试
    results = []
    
    results.append(("单客户端测试", test_single_client()))
    results.append(("管理器测试", test_client_manager()))
    results.append(("工作流测试", test_translation_workflow()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
            
    print()
    if all_passed:
        print("🎉 所有测试通过!")
    else:
        print("⚠️  部分测试失败")
        
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
