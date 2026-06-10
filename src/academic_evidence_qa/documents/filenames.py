"""模块名称：原文件名展示元数据。

用途：统一上传和显式补录的名称校验，不将客户端名称用于文件存储。
核心设计：只去除客户端目录前缀，保留可核验的 Unicode 名称及字面省略号。
工程要点：未知历史名称由调用方保留为 None，不从题名或对象路径推测。
"""

import unicodedata

from academic_evidence_qa.contracts.documents import DocumentError


def normalize_original_filename(filename: str) -> str:
    """名称：校验原文件名。用途：生成仅供展示和搜索的 basename。

    Args:
        filename: 上传声明或已核验导入记录中的名称，可能含客户端目录前缀。
    Returns:
        保留原有字符和空格的文件名，不含客户端目录前缀。
    Raises:
        DocumentError: 非字符串、无效 UTF-8、空白名、特殊目录名、控制字符或超长。
    """
    if not isinstance(filename, str):
        raise DocumentError("original_filename_invalid", 422)
    try:
        filename.encode("utf-8", "strict")
    except UnicodeEncodeError as error:
        raise DocumentError("original_filename_invalid", 422) from error
    if any(unicodedata.category(character) == "Cc" for character in filename):
        raise DocumentError("original_filename_invalid", 422)
    basename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    if not basename.strip() or basename.strip() in {".", ".."} or len(basename) > 1024:
        raise DocumentError("original_filename_invalid", 422)
    return basename
