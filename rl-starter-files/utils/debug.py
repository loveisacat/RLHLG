import traceback
import sys


def tracing():
    # 获取当前行的代码位置
    traceback.print_stack(limit=3)

    # 退出运行
    sys.exit()
