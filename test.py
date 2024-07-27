# def my_function():
#     """
#     这是一个示例函数的文档字符串。
#     在这里可以写关于函数的详细描述、参数说明、返回值等信息。
#     """
#     # 函数的实际代码
#
# print(my_function.__doc__)  # 打印函数的文档字符串
system_template = """
Use the following context to answer the user's question.
If you don't know the answer, say you don't, don't try to make it up. And answer in Chinese.
-----------
{question}
-----------
{chat_history}
"""