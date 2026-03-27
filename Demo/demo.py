# 测试脚本：查看 connect 函数的参数
"""from inspect import signature
from taosrest import connect

# 查看 connect 函数的参数签名
sig = signature(connect)
print("connect 函数参数签名：")
for param_name, param in sig.parameters.items():
    print(f"  {param_name}: {param}")
"""

# 查看 taosrest 库的版本和源代码
import taosrest
import inspect
import os

# 打印库版本
print(f"taosrest 版本: {taosrest.__version__ if hasattr(taosrest, '__version__') else '未知'}")

# 查找库的安装位置
print(f"\ntaosrest 安装路径: {taosrest.__file__}")

# 查看 connection 模块的 __init__ 方法
from taosrest.connection import TaosRestConnection

print("\nTaosRestConnection.__init__ 方法签名:")
try:
    # 获取 __init__ 方法的源代码
    source = inspect.getsource(TaosRestConnection.__init__)
    print("源代码:")
    print(source[:500])  # 只打印前500个字符
except Exception as e:
    print(f"无法获取源代码: {e}")

# 查看 __init__ 方法的参数
sig = inspect.signature(TaosRestConnection.__init__)
print(f"\n__init__ 方法参数:")
for param_name, param in sig.parameters.items():
    print(f"  {param_name}: {param}")

