from __future__ import annotations

import csv
import json
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from posixpath import normpath
from xml.etree import ElementTree


PARSE_WARNINGS: list[dict[str, str]] = []


class _PlainHtmlParser(HTMLParser):
    # Небольшой HTML-парсер без сторонних библиотек для извлечения текстового содержимого.
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self._parts.append(text)

    @property
    def text(self) -> str:
        return " ".join(self._parts)


def reset_parse_warnings() -> None:
    # Перед новым запуском модуля очищаем список предупреждений по проблемным файлам.
    PARSE_WARNINGS.clear()


def get_parse_warnings() -> list[dict[str, str]]:
    # Возвращаем копию предупреждений, чтобы агент мог сохранить их в отчет и CSV.
    return list(PARSE_WARNINGS)


def register_parse_warning(path: Path, warning_code: str, message: str) -> None:
    # Фиксируем случаи, когда файл принят в обработку, но прочитан через fallback-логику.
    PARSE_WARNINGS.append(
        {
            "file_path": str(path).replace("\\", "/"),
            "warning_code": warning_code,
            "message": message,
        }
    )


def openxml_fallback_text(path: Path, declared_format: str, warning_code: str, message: str) -> str:
    # Если современный Office-файл оказался некорректным, не прерываем работу:
    # регистрируем предупреждение и формируем безопасное текстовое описание источника.
    register_parse_warning(path, warning_code, message)
    binary_description = parse_binary_metadata(path)
    fallback = (
        f"Источник {path.name} заявлен как {declared_format}, но не является корректным файлом этого формата. "
        f"{message} Использован резервный режим обработки. {binary_description}"
    )
    return normalize_text(fallback)


def normalize_text(text: str) -> str:
    # Приводим текст к единому виду перед анализом и сохранением в датасет.
    return re.sub(r"\s+", " ", text, flags=re.MULTILINE).strip()


def parse_text_file(path: Path) -> str:
    return normalize_text(path.read_text(encoding="utf-8", errors="ignore"))


def parse_markdown(path: Path) -> str:
    content = re.sub(r"[#*_>`-]", " ", path.read_text(encoding="utf-8", errors="ignore"))
    return normalize_text(content)


def parse_html(path: Path) -> str:
    parser = _PlainHtmlParser()
    parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
    return normalize_text(parser.text)


def parse_json(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    parts: list[str] = []

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)
        elif isinstance(value, (str, int, float)):
            parts.append(str(value))

    walk(payload)
    return normalize_text(" ".join(parts))


def parse_csv_file(path: Path) -> str:
    rows: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        reader = csv.reader(file)
        for row in reader:
            rows.append(" ".join(row))
    return normalize_text(" ".join(rows))


def parse_docx(path: Path) -> str:
    # Извлекаем текст из docx напрямую как из ZIP-архива OpenXML.
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
        root = ElementTree.fromstring(xml)
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs: list[str] = []
        for paragraph in root.findall(".//w:p", namespace):
            parts = [node.text for node in paragraph.findall(".//w:t", namespace) if node.text]
            if parts:
                paragraphs.append("".join(parts))
        return normalize_text(" ".join(paragraphs))
    except zipfile.BadZipFile:
        return openxml_fallback_text(
            path,
            "DOCX",
            "bad_docx_zip",
            "Файл имеет расширение .docx, но не является корректным OpenXML-архивом. Вероятно, это старый .doc или поврежденный документ.",
        )
    except (KeyError, ElementTree.ParseError):
        return openxml_fallback_text(
            path,
            "DOCX",
            "docx_structure_error",
            "Структура DOCX повреждена или из нее невозможно извлечь текстовые узлы.",
        )


def parse_pptx(path: Path) -> str:
    # Извлекаем текст из презентации PowerPoint формата pptx как из OpenXML-архива.
    try:
        with zipfile.ZipFile(path) as archive:
            slide_names = sorted(
                name
                for name in archive.namelist()
                if name.startswith("ppt/slides/slide") and name.endswith(".xml")
            )
            namespace = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
            parts: list[str] = []
            for slide_name in slide_names:
                root = ElementTree.fromstring(archive.read(slide_name))
                slide_text = "".join(node.text or "" for node in root.findall(".//a:t", namespace))
                if slide_text.strip():
                    parts.append(slide_text.strip())
        return normalize_text(" ".join(parts))
    except zipfile.BadZipFile:
        return openxml_fallback_text(
            path,
            "PPTX",
            "bad_pptx_zip",
            "Файл имеет расширение .pptx, но не является корректным OpenXML-архивом. Вероятно, это старый .ppt или поврежденная презентация.",
        )
    except (KeyError, ElementTree.ParseError):
        return openxml_fallback_text(
            path,
            "PPTX",
            "pptx_structure_error",
            "Структура PPTX повреждена или из нее невозможно извлечь текстовые слайды.",
        )


def extract_office_media_descriptions(path: Path) -> list[str]:
    # Извлекаем сведения о встроенных изображениях и медиа из Office OpenXML-файлов.
    # Это не OCR, но позволяет учитывать все вложенные картинки, схемы и иллюстрации в аналитике.
    extension = path.suffix.lower()
    media_prefix = {
        ".docx": "word/media/",
        ".pptx": "ppt/media/",
        ".xlsx": "xl/media/",
        ".xlsm": "xl/media/",
    }.get(extension)
    if media_prefix is None:
        return []
    try:
        with zipfile.ZipFile(path) as archive:
            media_names = sorted(
                name for name in archive.namelist() if name.startswith(media_prefix) and not name.endswith("/")
            )
    except (zipfile.BadZipFile, KeyError):
        return []
    descriptions: list[str] = []
    for index, media_name in enumerate(media_names, start=1):
        media_file = Path(media_name)
        descriptions.append(
            normalize_text(
                f"Встроенное изображение или медиаобъект {index} в файле {path.name}: "
                f"{media_file.name}, формат {media_file.suffix.lower().lstrip('.') or 'unknown'}."
            )
        )
    return descriptions


def extract_media_descriptions(path: Path) -> list[str]:
    # Единая точка учета медиа: внешние изображения/видео и встроенные картинки внутри офисных документов.
    extension = path.suffix.lower()
    if extension in {".docx", ".pptx", ".xlsx", ".xlsm"}:
        return extract_office_media_descriptions(path)
    if extension in {".png", ".jpg", ".jpeg", ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm", ".svg"}:
        return [extract_text(path)]
    return []


def parse_xlsx(path: Path) -> str:
    # Читаем Excel-файлы формата xlsx/xlsm без pandas и openpyxl, чтобы решение оставалось офлайн и простым.
    try:
        with zipfile.ZipFile(path) as archive:
            shared_strings: list[str] = []
            if "xl/sharedStrings.xml" in archive.namelist():
                root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
                namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                for item in root.findall("a:si", namespace):
                    text = "".join(node.text or "" for node in item.findall(".//a:t", namespace))
                    shared_strings.append(text)

            workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
            rels = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            ns = {
                "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
                "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
                "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
            }
            rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall("pr:Relationship", ns)}
            chunks: list[str] = []
            for sheet in workbook.findall("a:sheets/a:sheet", ns):
                sheet_name = sheet.attrib["name"]
                target = rel_map[sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]]
                sheet_path = normpath(f"xl/{target}").replace("\\", "/")
                sheet_root = ElementTree.fromstring(archive.read(sheet_path))
                chunks.append(sheet_name)
                for row in sheet_root.findall("a:sheetData/a:row", ns):
                    values: list[str] = []
                    for cell in row.findall("a:c", ns):
                        cell_type = cell.attrib.get("t")
                        value = ""
                        if cell_type == "s":
                            value_node = cell.find("a:v", ns)
                            if value_node is not None and value_node.text is not None:
                                index = int(value_node.text)
                                if 0 <= index < len(shared_strings):
                                    value = shared_strings[index]
                        elif cell_type == "inlineStr":
                            value = "".join(node.text or "" for node in cell.findall(".//a:t", ns))
                        elif cell_type == "b":
                            value_node = cell.find("a:v", ns)
                            if value_node is not None and value_node.text is not None:
                                value = "TRUE" if value_node.text == "1" else "FALSE"
                        else:
                            value_node = cell.find("a:v", ns)
                            if value_node is not None and value_node.text is not None:
                                value = value_node.text
                            formula_node = cell.find("a:f", ns)
                            if formula_node is not None and formula_node.text:
                                value = f"{value} [formula:{formula_node.text}]".strip()
                        if value.strip():
                            values.append(value.strip())
                    if values:
                        chunks.append(" ".join(values))
        return normalize_text(" ".join(chunks))
    except zipfile.BadZipFile:
        return openxml_fallback_text(
            path,
            "XLSX/XLSM",
            "bad_xlsx_zip",
            "Файл имеет расширение .xlsx/.xlsm, но не является корректным OpenXML-архивом. Вероятно, это старый .xls или поврежденная книга.",
        )
    except (KeyError, ElementTree.ParseError):
        return openxml_fallback_text(
            path,
            "XLSX/XLSM",
            "xlsx_structure_error",
            "Структура книги Excel повреждена или из нее невозможно извлечь ячейки и листы.",
        )


def parse_pdf(path: Path) -> str:
    # Для PDF используем легкий офлайн-разбор текста без внешних зависимостей.
    binary = path.read_bytes()
    text_chunks: list[str] = []
    for match in re.finditer(rb"\(([^()]*)\)", binary):
        chunk = match.group(1).decode("latin-1", errors="ignore")
        chunk = chunk.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
        if any(symbol.isalpha() for symbol in chunk):
            text_chunks.append(chunk)
    if text_chunks:
        return normalize_text(" ".join(text_chunks))
    return normalize_text(f"PDF документ {path.name}. Текст извлечен частично, требуется sidecar-описание при сложной верстке.")


def parse_svg(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    title = re.search(r"<title>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
    desc = re.search(r"<desc>(.*?)</desc>", text, flags=re.IGNORECASE | re.DOTALL)
    return normalize_text(f"{title.group(1) if title else ''}. {desc.group(1) if desc else ''}")


def parse_binary_metadata(path: Path) -> str:
    # Для фото и видео читаем sidecar-описание, если пользователь его положил рядом.
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    if sidecar.exists():
        return parse_json(sidecar)
    size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0
    readable_name = path.stem.replace("_", " ").replace("-", " ").strip()
    media_type = "Видеофайл" if path.suffix.lower() in {".mp4", ".avi", ".mov"} else "Изображение"
    return normalize_text(f"{media_type} {path.name}. Предполагаемая тема: {readable_name}. Размер файла: {size_mb:.2f} МБ.")


def extract_text(path: Path) -> str:
    # Единая точка выбора парсера по расширению файла.
    extension = path.suffix.lower()
    if extension == ".txt":
        return parse_text_file(path)
    if extension in {".doc", ".xls"}:
        return parse_binary_metadata(path)
    if extension == ".md":
        return parse_markdown(path)
    if extension in {".html", ".htm"}:
        return parse_html(path)
    if extension == ".json":
        return parse_json(path)
    if extension == ".csv":
        return parse_csv_file(path)
    if extension == ".docx":
        return parse_docx(path)
    if extension == ".pptx":
        return parse_pptx(path)
    if extension in {".xlsx", ".xlsm"}:
        return parse_xlsx(path)
    if extension == ".pdf":
        return parse_pdf(path)
    if extension == ".svg":
        return parse_svg(path)
    if extension in {".png", ".jpg", ".jpeg", ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm"}:
        return parse_binary_metadata(path)
    return parse_text_file(path)
