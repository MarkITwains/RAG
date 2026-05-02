#!/usr/bin/env python3
"""
文档预处理脚本 - 批量修复编码和清理OCR乱码

功能：
1. 自动检测并修复文件编码（GBK/GB2312/UTF-8等）
2. 清理OCR识别产生的乱码字符
3. 识别并标注GB/T标准文档的章节结构
4. 输出清洗后的UTF-8编码文件

用法：
    python preprocess_docs.py [--input-dir INPUT] [--output-dir OUTPUT] [--dry-run]
"""

import os
import re
import sys
import argparse
import shutil
from pathlib import Path
from typing import Optional, Tuple, List

# 尝试导入chardet，如果没有则使用简单的编码检测
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False
    print("⚠️  chardet未安装，将使用简化的编码检测（pip install chardet 可获得更好效果）")


# ================== 编码检测与转换 ==================

def detect_encoding_simple(data: bytes) -> str:
    """简化的编码检测（无chardet时使用）"""
    # 检查BOM
    if data.startswith(b'\xef\xbb\xbf'):
        return 'utf-8-sig'
    if data.startswith(b'\xff\xfe'):
        return 'utf-16-le'
    if data.startswith(b'\xfe\xff'):
        return 'utf-16-be'
    
    # 尝试UTF-8
    try:
        data.decode('utf-8')
        return 'utf-8'
    except UnicodeDecodeError:
        pass
    
    # 尝试GBK（中文Windows常见）
    try:
        decoded = data.decode('gbk')
        # 验证是否包含合理的中文字符
        chinese_ratio = len(re.findall(r'[\u4e00-\u9fff]', decoded)) / max(len(decoded), 1)
        if chinese_ratio > 0.1:  # 超过10%是中文
            return 'gbk'
    except UnicodeDecodeError:
        pass
    
    # 尝试GB2312
    try:
        data.decode('gb2312')
        return 'gb2312'
    except UnicodeDecodeError:
        pass
    
    # 尝试GB18030（最全的中文编码）
    try:
        data.decode('gb18030')
        return 'gb18030'
    except UnicodeDecodeError:
        pass
    
    # 默认使用latin-1（不会报错）
    return 'latin-1'


def detect_encoding(data: bytes) -> Tuple[str, float]:
    """检测字节数据的编码"""
    # 首先尝试UTF-8（最常见且最安全）
    try:
        decoded = data.decode('utf-8')
        # 验证是否有合理的中文内容
        chinese_ratio = len(re.findall(r'[\u4e00-\u9fff]', decoded)) / max(len(decoded), 1)
        garbage_ratio = calculate_garbage_ratio(decoded)
        
        # 如果UTF-8解码成功且质量好，直接返回
        if garbage_ratio < 0.05 and chinese_ratio > 0.01:
            return 'utf-8', 0.99
    except UnicodeDecodeError:
        pass
    
    if HAS_CHARDET:
        result = chardet.detect(data)
        encoding = result.get('encoding', 'utf-8') or 'utf-8'
        confidence = result.get('confidence', 0.0) or 0.0
        
        # chardet有时会误判UTF-8为GB2312
        if encoding.lower() in ('gb2312', 'gbk', 'gb18030'):
            # 再次验证UTF-8
            try:
                utf8_decoded = data.decode('utf-8')
                utf8_garbage = calculate_garbage_ratio(utf8_decoded)
                
                # 如果UTF-8质量更好，使用UTF-8
                if utf8_garbage < 0.05:
                    return 'utf-8', 0.99
            except:
                pass
        
        # chardet有时会误判中文为ISO-8859-1
        if encoding.lower() in ('iso-8859-1', 'ascii', 'windows-1252') and confidence < 0.9:
            # 尝试GBK
            try:
                decoded = data.decode('gbk')
                chinese_count = len(re.findall(r'[\u4e00-\u9fff]', decoded))
                if chinese_count > 10:  # 有较多中文
                    return 'gbk', 0.95
            except:
                pass
        
        return encoding, confidence
    else:
        return detect_encoding_simple(data), 0.8


def decode_file_content(file_path: str) -> Tuple[str, str]:
    """读取文件并尝试正确解码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    
    # 检测编码
    encoding, confidence = detect_encoding(raw_data)
    
    # 如果检测为UTF-8，直接返回
    if encoding.lower() in ('utf-8', 'utf-8-sig'):
        try:
            return raw_data.decode('utf-8'), 'utf-8'
        except:
            pass
    
    # 尝试解码
    encodings_to_try = [encoding]
    
    # 添加备选编码
    if encoding.lower() not in ('utf-8', 'utf-8-sig'):
        encodings_to_try.append('utf-8')
    if encoding.lower() not in ('gbk', 'gb2312', 'gb18030'):
        encodings_to_try.extend(['gbk', 'gb18030', 'gb2312'])
    
    best_result = None
    best_score = float('inf')
    
    for enc in encodings_to_try:
        try:
            decoded = raw_data.decode(enc)
            # 验证解码质量
            score = score_text_quality(decoded)
            
            # 选择质量最好的解码结果
            if score < best_score:
                best_score = score
                best_result = (decoded, enc)
            
            # 如果找到高质量结果，提前返回
            if score < 0.05:
                return decoded, enc
        except (UnicodeDecodeError, LookupError):
            continue
    
    if best_result:
        repaired = try_mojibake_repair(best_result[0])
        if repaired and repaired[2] + 0.02 < best_score:
            return repaired[0], repaired[1]
    
    if best_result and best_score < 0.3:
        return best_result
    
    # 最后用errors='replace'强制解码
    return raw_data.decode('utf-8', errors='replace'), 'utf-8(forced)'


# ================== 乱码检测与清理 ==================

# 常见OCR乱码字符模式
GARBAGE_PATTERNS = [
    r'[ӡưճƬ֧أ͵һڱ]+',  # 常见的OCR乱码序列
    r'[ͶͶͶĄIJͶ]+',  # 希腊/扩展拉丁乱码
    r'[\x00-\x08\x0b\x0c\x0e-\x1f]',  # 控制字符
    r'(?<![a-zA-Z])[ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]{3,}(?![a-zA-Z])',  # 连续扩展拉丁字符
]

# 合法的Unicode范围（中文、日文、韩文、标点、数学符号等）
VALID_RANGES = [
    (0x0000, 0x007F),   # ASCII
    (0x00A0, 0x00FF),   # 拉丁补充（有限允许）
    (0x2000, 0x206F),   # 通用标点
    (0x2070, 0x209F),   # 上下标
    (0x20A0, 0x20CF),   # 货币符号
    (0x2100, 0x214F),   # 字母符号
    (0x2150, 0x218F),   # 数字形式
    (0x2190, 0x21FF),   # 箭头
    (0x2200, 0x22FF),   # 数学运算符
    (0x2300, 0x23FF),   # 杂项技术符号
    (0x2460, 0x24FF),   # 带圈字母数字
    (0x2500, 0x257F),   # 制表符
    (0x25A0, 0x25FF),   # 几何形状
    (0x2600, 0x26FF),   # 杂项符号
    (0x3000, 0x303F),   # CJK标点
    (0x3040, 0x309F),   # 平假名
    (0x30A0, 0x30FF),   # 片假名
    (0x3100, 0x312F),   # 注音符号
    (0x3200, 0x32FF),   # 带圈CJK
    (0x3400, 0x4DBF),   # CJK扩展A
    (0x4E00, 0x9FFF),   # CJK基本
    (0xF900, 0xFAFF),   # CJK兼容
    (0xFE30, 0xFE4F),   # CJK兼容形式
    (0xFF00, 0xFFEF),   # 半角全角形式
]


def is_valid_char(char: str) -> bool:
    """检查字符是否在合法Unicode范围内"""
    code = ord(char)
    for start, end in VALID_RANGES:
        if start <= code <= end:
            return True
    return False


def calculate_garbage_ratio(text: str) -> float:
    """计算文本中乱码字符的比例"""
    if not text:
        return 0.0
    
    total = len(text)
    garbage_count = 0
    
    for char in text:
        if char.isspace():
            continue
        if char == '\ufffd':  # Unicode replacement char
            garbage_count += 1
            continue
        if not is_valid_char(char):
            garbage_count += 1
    
    return garbage_count / total if total > 0 else 0.0


def calculate_chinese_ratio(text: str) -> float:
    """计算中文字符比例"""
    if not text:
        return 0.0
    chinese_count = len(re.findall(r'[\u4e00-\u9fff]', text))
    return chinese_count / max(len(text), 1)


def calculate_replacement_ratio(text: str) -> float:
    """计算替换字符(�)比例"""
    if not text:
        return 0.0
    return text.count('\ufffd') / max(len(text), 1)


def calculate_mojibake_ratio(text: str) -> float:
    """计算疑似乱码(扩展拉丁)比例"""
    if not text:
        return 0.0
    mojibake_count = 0
    for char in text:
        code = ord(char)
        if 0x00C0 <= code <= 0x00FF:
            mojibake_count += 1
    return mojibake_count / max(len(text), 1)


def score_text_quality(text: str) -> float:
    """文本质量评分（越低越好）"""
    garbage_ratio = calculate_garbage_ratio(text)
    replacement_ratio = calculate_replacement_ratio(text)
    mojibake_ratio = calculate_mojibake_ratio(text)
    chinese_ratio = calculate_chinese_ratio(text)

    score = garbage_ratio + replacement_ratio * 2.0 + mojibake_ratio * 0.6

    # 中文文档倾向性：中文比例低时稍作惩罚
    if len(text) > 50 and chinese_ratio < 0.01:
        score += 0.05
    else:
        score -= min(chinese_ratio, 0.2) * 0.5

    return score


def should_try_mojibake_repair(text: str) -> bool:
    """判断是否需要尝试乱码逆修复"""
    if not text or len(text) < 50:
        return False
    replacement_ratio = calculate_replacement_ratio(text)
    mojibake_ratio = calculate_mojibake_ratio(text)
    chinese_ratio = calculate_chinese_ratio(text)

    return replacement_ratio > 0.002 or (mojibake_ratio > 0.08 and chinese_ratio < 0.02)


def try_mojibake_repair(text: str) -> Optional[Tuple[str, str, float]]:
    """尝试对常见乱码进行逆修复，返回(修复文本, 修复方式, 评分)"""
    if not should_try_mojibake_repair(text):
        return None

    candidates: List[Tuple[float, str, str]] = []
    source_encodings = ['latin1', 'cp1252']
    target_encodings = ['utf-8', 'gb18030', 'gbk']

    for src in source_encodings:
        try:
            raw = text.encode(src)
        except UnicodeEncodeError:
            continue
        for dst in target_encodings:
            try:
                decoded = raw.decode(dst)
            except UnicodeDecodeError:
                continue
            score = score_text_quality(decoded)
            candidates.append((score, decoded, f"mojibake:{src}->{dst}"))

    if not candidates:
        return None

    best = min(candidates, key=lambda x: x[0])
    return best[1], best[2], best[0]


def clean_garbage_chars(text: str) -> str:
    """清理乱码字符"""
    if not text:
        return ""
    
    # 1. 移除控制字符和替换字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    text = text.replace('\ufffd', '')
    
    # 2. 移除连续的乱码序列
    for pattern in GARBAGE_PATTERNS:
        text = re.sub(pattern, '', text)
    
    # 3. 逐字符过滤非法Unicode
    cleaned_chars = []
    for char in text:
        if char.isspace() or is_valid_char(char):
            cleaned_chars.append(char)
        # else: 丢弃非法字符
    
    text = ''.join(cleaned_chars)
    
    # 4. 清理多余空白
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)
    
    return text.strip()


# ================== 结构识别与标注 ==================

# GB/T标准文档章节模式
SECTION_PATTERNS = [
    # 主章节：1 范围、2 术语、3 要求 等
    (r'^(\d+)\s+([^\n]{2,50})$', 'chapter'),
    # 二级：1.1 一般要求
    (r'^(\d+\.\d+)\s+([^\n]{2,50})$', 'section'),
    # 三级：1.1.1 xxx
    (r'^(\d+\.\d+\.\d+)\s+([^\n]{2,50})$', 'subsection'),
    # 四级及以上：1.1.1.1 xxx
    (r'^(\d+(?:\.\d+){3,})\s+([^\n]{2,40})$', 'subsubsection'),
    # 附录：附录A、附录 B
    (r'^(附录\s*[A-Z])\s*([^\n]*)$', 'appendix'),
    # 表：表1、表 A.1
    (r'^(表\s*[\dA-Z]+(?:\.\d+)?)\s*(.*)$', 'table'),
    # 图：图1、图 A.1
    (r'^(图\s*[\dA-Z]+(?:\.\d+)?)\s*(.*)$', 'figure'),
]


def identify_sections(text: str) -> List[Tuple[int, int, str, str, str]]:
    """识别文档中的章节结构
    
    返回: [(start_pos, end_pos, section_num, section_title, section_type), ...]
    """
    sections = []
    lines = text.split('\n')
    
    current_pos = 0
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        for pattern, section_type in SECTION_PATTERNS:
            match = re.match(pattern, line_stripped)
            if match:
                section_num = match.group(1)
                section_title = match.group(2) if match.lastindex >= 2 else ""
                sections.append((current_pos, current_pos + len(line), section_num, section_title.strip(), section_type))
                break
        
        current_pos += len(line) + 1  # +1 for newline
    
    return sections


def add_section_markers(text: str) -> str:
    """为章节添加明确的标记，便于切块时识别"""
    sections = identify_sections(text)
    
    if not sections:
        return text
    
    # 按位置倒序处理，避免位置偏移
    result = text
    for start, end, num, title, stype in reversed(sections):
        marker = f"\n\n<!-- SECTION: {stype} | {num} | {title} -->\n"
        # 在章节标题前插入标记
        result = result[:start] + marker + result[start:]
    
    return result


# ================== 主处理逻辑 ==================

def process_file(input_path: str, output_path: str, dry_run: bool = False) -> dict:
    """处理单个文件"""
    result = {
        'input': input_path,
        'output': output_path,
        'original_encoding': None,
        'garbage_ratio_before': 0.0,
        'garbage_ratio_after': 0.0,
        'sections_found': 0,
        'status': 'unknown',
        'message': ''
    }
    
    try:
        # 1. 读取并解码
        content, encoding = decode_file_content(input_path)
        result['original_encoding'] = encoding
        
        # 2. 计算原始乱码比例
        result['garbage_ratio_before'] = calculate_garbage_ratio(content)
        quality_score_before = score_text_quality(content)
        
        # 如果原始内容质量已经很好，且是UTF-8，跳过处理
        if (
            encoding.lower() in ('utf-8', 'utf-8-sig')
            and quality_score_before < 0.05
            and not should_try_mojibake_repair(content)
        ):
            result['status'] = 'skip'
            result['message'] = f"编码: {encoding}, 质量良好，无需处理"
            return result
        
        # 3. 清理乱码
        cleaned = clean_garbage_chars(content)
        result['garbage_ratio_after'] = calculate_garbage_ratio(cleaned)
        
        # 4. 识别章节结构
        sections = identify_sections(cleaned)
        result['sections_found'] = len(sections)
        
        # 5. 验证处理质量
        if result['garbage_ratio_after'] > result['garbage_ratio_before'] + 0.01:
            # 如果处理后质量变差，使用原始内容
            result['status'] = 'skip'
            result['message'] = f"编码: {encoding}, 处理后质量下降，保持原样"
            return result
        
        # 6. 写入输出
        if not dry_run:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned)
        
        result['status'] = 'success'
        result['message'] = f"编码: {encoding}, 乱码率: {result['garbage_ratio_before']:.1%} → {result['garbage_ratio_after']:.1%}, 章节: {len(sections)}"
        
    except Exception as e:
        result['status'] = 'error'
        result['message'] = str(e)
    
    return result


def process_directory(input_dir: str, output_dir: str, dry_run: bool = False) -> List[dict]:
    """处理目录下所有文本文件"""
    results = []
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 支持的文件扩展名
    extensions = {'.txt', '.md', '.text'}
    
    for file_path in input_path.rglob('*'):
        if file_path.suffix.lower() not in extensions:
            continue
        
        # 计算相对路径和输出路径
        rel_path = file_path.relative_to(input_path)
        out_file = output_path / rel_path
        
        print(f"处理: {rel_path}...", end=' ')
        result = process_file(str(file_path), str(out_file), dry_run)
        results.append(result)
        
        status_icon = '✅' if result['status'] == 'success' else '❌'
        print(f"{status_icon} {result['message']}")
    
    return results


def print_summary(results: List[dict]):
    """打印处理摘要"""
    total = len(results)
    success = sum(1 for r in results if r['status'] == 'success')
    
    encoding_stats = {}
    for r in results:
        enc = r.get('original_encoding', 'unknown')
        encoding_stats[enc] = encoding_stats.get(enc, 0) + 1
    
    high_garbage = [r for r in results if r.get('garbage_ratio_before', 0) > 0.1]
    
    print("\n" + "=" * 60)
    print("📊 处理摘要")
    print("=" * 60)
    print(f"总文件数: {total}")
    print(f"成功处理: {success}")
    print(f"处理失败: {total - success}")
    
    print(f"\n📝 编码分布:")
    for enc, count in sorted(encoding_stats.items(), key=lambda x: -x[1]):
        print(f"  - {enc}: {count} 个文件")
    
    if high_garbage:
        print(f"\n⚠️  高乱码率文件 (>10%):")
        for r in high_garbage[:10]:
            print(f"  - {Path(r['input']).name}: {r['garbage_ratio_before']:.1%}")
        if len(high_garbage) > 10:
            print(f"  ... 还有 {len(high_garbage) - 10} 个文件")


def main():
    parser = argparse.ArgumentParser(description='文档预处理 - 修复编码和清理乱码')
    parser.add_argument('--input-dir', '-i', default='./data/clear_docs',
                       help='输入目录 (默认: ./data/clear_docs)')
    parser.add_argument('--output-dir', '-o', default='./data/processed_docs',
                       help='输出目录 (默认: ./data/processed_docs)')
    parser.add_argument('--dry-run', '-n', action='store_true',
                       help='仅检测，不实际写入文件')
    parser.add_argument('--file', '-f', 
                       help='仅处理单个文件')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📄 文档预处理工具")
    print("=" * 60)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"模式: {'检测模式（不写入）' if args.dry_run else '处理模式'}")
    print("=" * 60 + "\n")
    
    if args.file:
        # 处理单个文件
        out_file = Path(args.output_dir) / Path(args.file).name
        result = process_file(args.file, str(out_file), args.dry_run)
        print(f"结果: {result['status']} - {result['message']}")
    else:
        # 处理整个目录
        results = process_directory(args.input_dir, args.output_dir, args.dry_run)
        print_summary(results)
    
    print("\n✅ 处理完成!")


if __name__ == '__main__':
    main()
