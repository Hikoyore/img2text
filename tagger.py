#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image to Text для Stable Diffusion
Версия: 3.0
Автор: Hikoyore
Описание: Полнофункциональный теггер изображений с использованием моделей WD14.
Поддерживает пакетную обработку, различные форматы вывода (включая Automatic1111),
кэширование, чёрные списки, историю, локализацию, автообновление моделей и многое другое.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import threading
import os
import json
import logging
import time
import glob
import hashlib
import webbrowser
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Set
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import requests
from huggingface_hub import hf_hub_download, HfApi
import onnxruntime as ort

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False
    TkinterDnD = tk.Tk

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

MODELS = {
    "ViT (быстрый)": "SmilingWolf/wd-v1-4-vit-tagger-v2",
    "ConvNext (средний)": "SmilingWolf/wd-v1-4-convnext-tagger-v2",
    "SwinV2 (точный)": "SmilingWolf/wd-swinv2-tagger-v3"
}
DEFAULT_MODEL = "ConvNext (средний)"

ONNX_FILE = "model.onnx"
TAGS_FILE = "selected_tags.csv"
CONFIG_FILE = "tagger_config.json"
TRANSLATIONS_FILE = "translations.json"
BLACKLIST_FILE = "blacklist.txt"
WHITELIST_FILE = "whitelist.txt"
CACHE_DIR = "cache"

IMAGE_SIZE = 448

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tagger.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

PROVIDERS = []
if 'CUDAExecutionProvider' in ort.get_available_providers():
    PROVIDERS.append('CUDAExecutionProvider')
PROVIDERS.append('CPUExecutionProvider')
DEVICE = 'cuda' if 'CUDAExecutionProvider' in PROVIDERS else 'cpu'
logging.info(f"Доступные провайдеры ONNX: {PROVIDERS}")

def load_model_and_tags(model_key: str) -> Tuple[ort.InferenceSession, List[str], List[int], List[int], List[int], List[int], List[int], List[int]]:
    repo_id = MODELS[model_key]
    logging.info(f"Загрузка модели из {repo_id}...")

    onnx_path = hf_hub_download(repo_id=repo_id, filename=ONNX_FILE, local_dir=".")
    logging.info(f"ONNX модель загружена: {onnx_path}")

    session = ort.InferenceSession(onnx_path, providers=PROVIDERS)
    logging.info("Сессия ONNX создана.")

    if not os.path.exists(TAGS_FILE):
        logging.info("Скачивание файла тегов...")
        hf_hub_download(repo_id=repo_id, filename=TAGS_FILE, local_dir=".")
        logging.info("Файл тегов скачан.")

    tags_df = pd.read_csv(TAGS_FILE)

    if 'category' in tags_df.columns:
        rating_tags = tags_df[tags_df['category'] == 9]['name'].tolist()
        general_tags = tags_df[tags_df['category'] == 0]['name'].tolist()
        character_tags = tags_df[tags_df['category'] == 4]['name'].tolist()
        artist_tags = tags_df[tags_df['category'] == 1]['name'].tolist()
        copyright_tags = tags_df[tags_df['category'] == 3]['name'].tolist()
        other_tags = tags_df[~tags_df['category'].isin([0,1,3,4,9])]['name'].tolist()
    else:
        if 'name' in tags_df.columns:
            all_tags = tags_df['name'].tolist()
        else:
            all_tags = tags_df.iloc[:, 0].tolist()
        rating_tags = []
        general_tags = all_tags
        character_tags = []
        artist_tags = []
        copyright_tags = []
        other_tags = []
        logging.warning("Колонка 'category' не найдена, все теги считаются общими.")

    all_tags = (rating_tags + general_tags + character_tags + artist_tags + copyright_tags + other_tags)
    category_map = [9] * len(rating_tags) + [0] * len(general_tags) + [4] * len(character_tags) + \
                   [1] * len(artist_tags) + [3] * len(copyright_tags) + [5] * len(other_tags)

    rating_indices = list(range(len(rating_tags)))
    general_indices = list(range(len(rating_tags), len(rating_tags) + len(general_tags)))
    character_indices = list(range(len(rating_tags) + len(general_tags),
                                   len(rating_tags) + len(general_tags) + len(character_tags)))
    artist_indices = list(range(len(rating_tags) + len(general_tags) + len(character_tags),
                                len(rating_tags) + len(general_tags) + len(character_tags) + len(artist_tags)))
    copyright_indices = list(range(len(rating_tags) + len(general_tags) + len(character_tags) + len(artist_tags),
                                   len(rating_tags) + len(general_tags) + len(character_tags) + len(artist_tags) + len(copyright_tags)))
    other_indices = list(range(len(rating_tags) + len(general_tags) + len(character_tags) + len(artist_tags) + len(copyright_tags),
                               len(all_tags)))

    logging.info(f"Загружено тегов: всего {len(all_tags)} (rating: {len(rating_tags)}, general: {len(general_tags)}, "
                 f"character: {len(character_tags)}, artist: {len(artist_tags)}, copyright: {len(copyright_tags)}, other: {len(other_tags)})")

    return session, all_tags, rating_indices, general_indices, character_indices, artist_indices, copyright_indices, other_indices

def mcut_threshold(scores: np.ndarray, edge_ratio: float = 0.9, min_ratio: float = 0.1) -> float:
    if len(scores) == 0:
        return 0.5
    sorted_scores = np.sort(scores)[::-1]
    diffs = np.diff(sorted_scores)
    if len(diffs) == 0:
        return sorted_scores[0] * 0.5
    normalized_diffs = diffs / np.max(diffs)
    threshold_idx = np.argmax(normalized_diffs >= edge_ratio) if np.any(normalized_diffs >= edge_ratio) else np.argmax(diffs)
    candidate = sorted_scores[threshold_idx + 1] if threshold_idx + 1 < len(sorted_scores) else sorted_scores[-1]
    return np.clip(candidate, min_ratio, 0.95)

def preprocess_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert('RGB')
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    img_array = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_array = img_array[np.newaxis, ...]
    return img_array

def ensure_dir(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_string_list(filepath: str) -> Set[str]:
    if not os.path.exists(filepath):
        return set()
    with open(filepath, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}

def save_string_list(filepath: str, items: Set[str]):
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in sorted(items):
            f.write(item + '\n')

class Cache:
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self._cache = {}
        self._access = []

    def get(self, key: str):
        if key in self._cache:
            self._access.remove(key)
            self._access.append(key)
            return self._cache[key]
        return None

    def set(self, key: str, value):
        if key in self._cache:
            self._cache[key] = value
            self._access.remove(key)
            self._access.append(key)
        else:
            if len(self._access) >= self.maxsize:
                oldest = self._access.pop(0)
                del self._cache[oldest]
            self._cache[key] = value
            self._access.append(key)

    def clear(self):
        self._cache.clear()
        self._access.clear()

class TaggerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image to Text for Stable Diffusion")
        self.root.geometry("1200x900")

        self.image_path = None
        self.current_image = None
        self.current_photo = None
        self.history = deque(maxlen=20)
        self.cache = Cache(maxsize=50)

        self.current_model_key = DEFAULT_MODEL
        self.session = None
        self.all_tags = []
        self.rating_idx = []
        self.general_idx = []
        self.character_idx = []
        self.artist_idx = []
        self.copyright_idx = []
        self.other_idx = []
        self.load_model_in_thread()

        self.general_threshold = tk.DoubleVar(value=0.35)
        self.character_threshold = tk.DoubleVar(value=0.85)
        self.artist_threshold = tk.DoubleVar(value=0.75)
        self.copyright_threshold = tk.DoubleVar(value=0.75)
        self.rating_threshold = tk.DoubleVar(value=0.5)
        self.use_mcut = tk.BooleanVar(value=False)
        self.mcut_edge_ratio = tk.DoubleVar(value=0.9)
        self.mcut_min_ratio = tk.DoubleVar(value=0.1)
        self.remove_overlap = tk.BooleanVar(value=True)
        self.output_format = tk.StringVar(value="a1111")
        self.a1111_prefix = tk.StringVar(value="")
        self.a1111_suffix = tk.StringVar(value="")
        self.group_by_category = tk.BooleanVar(value=False)
        self.use_blacklist = tk.BooleanVar(value=False)
        self.use_whitelist = tk.BooleanVar(value=False)
        self.blacklist = set()
        self.whitelist = set()
        self.lang = tk.StringVar(value="ru")
        self.check_updates = tk.BooleanVar(value=True)
        self.profiling = tk.BooleanVar(value=False)
        self.a1111_url = tk.StringVar(value="http://127.0.0.1:7860")
        self.a1111_enabled = tk.BooleanVar(value=False)

        self.translations = self.load_translations()
        self.current_strings = self.translations.get(self.lang.get(), self.translations['ru'])

        self.load_config()

        self.create_widgets()

        if HAS_DND:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.on_drop)

        self.root.bind('<F1>', self.show_help)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_translations(self) -> Dict:
        default = {
            "ru": {
                "open_image": "Открыть изображение",
                "batch_process": "Пакетная обработка",
                "generate": "Сгенерировать",
                "copy": "Копировать",
                "save": "Сохранить",
                "settings": "Настройки",
                "general_threshold": "Порог (общие)",
                "character_threshold": "Порог (персонажи)",
                "artist_threshold": "Порог (художники)",
                "copyright_threshold": "Порог (копирайт)",
                "rating_threshold": "Порог (рейтинг)",
                "use_mcut": "Использовать MCut",
                "mcut_edge_ratio": "Коэф. излома MCut",
                "mcut_min_ratio": "Мин. порог MCut",
                "remove_overlap": "Удалять пересекающиеся теги",
                "output_format": "Формат вывода",
                "simple": "Простой список",
                "with_probs": "С вероятностями",
                "a1111": "Automatic1111",
                "grouped": "Группировка",
                "a1111_prefix": "Префикс для A1111",
                "a1111_suffix": "Суффикс для A1111",
                "group_by_category": "Группировать по категориям",
                "blacklist": "Чёрный список",
                "whitelist": "Белый список",
                "use_blacklist": "Использовать чёрный список",
                "use_whitelist": "Использовать белый список",
                "load_blacklist": "Загрузить чёрный список",
                "load_whitelist": "Загрузить белый список",
                "history": "История",
                "clear_history": "Очистить историю",
                "model": "Модель",
                "check_updates": "Проверять обновления модели",
                "profiling": "Профилирование",
                "a1111_integration": "Интеграция с Automatic1111",
                "a1111_url": "URL API",
                "a1111_enabled": "Отправлять в A1111",
                "send_to_a1111": "Отправить в A1111",
                "language": "Язык",
                "help": "Справка",
                "exit": "Выход",
                "ready": "Готов к работе",
                "loading_model": "Загрузка модели...",
                "model_loaded": "Модель загружена",
                "processing": "Обработка...",
                "done": "Готово",
                "error": "Ошибка",
                "warning": "Предупреждение",
                "info": "Информация",
                "no_image": "Нет изображения",
                "select_image_first": "Сначала выберите изображение",
                "select_folder": "Выберите папку",
                "processing_cancelled": "Обработка отменена",
                "processing_complete": "Обработка завершена",
                "save_tags": "Сохранить теги",
                "copy_tags": "Копировать теги",
                "paste": "Вставить",
            },
            "en": {
                "open_image": "Open image",
                "batch_process": "Batch process",
                "generate": "Generate",
                "copy": "Copy",
                "save": "Save",
                "settings": "Settings",
                "general_threshold": "General threshold",
                "character_threshold": "Character threshold",
                "artist_threshold": "Artist threshold",
                "copyright_threshold": "Copyright threshold",
                "rating_threshold": "Rating threshold",
                "use_mcut": "Use MCut",
                "mcut_edge_ratio": "MCut edge ratio",
                "mcut_min_ratio": "MCut min ratio",
                "remove_overlap": "Remove overlapping tags",
                "output_format": "Output format",
                "simple": "Simple list",
                "with_probs": "With probabilities",
                "a1111": "Automatic1111",
                "grouped": "Grouped",
                "a1111_prefix": "A1111 prefix",
                "a1111_suffix": "A1111 suffix",
                "group_by_category": "Group by category",
                "blacklist": "Blacklist",
                "whitelist": "Whitelist",
                "use_blacklist": "Use blacklist",
                "use_whitelist": "Use whitelist",
                "load_blacklist": "Load blacklist",
                "load_whitelist": "Load whitelist",
                "history": "History",
                "clear_history": "Clear history",
                "model": "Model",
                "check_updates": "Check model updates",
                "profiling": "Profiling",
                "a1111_integration": "Automatic1111 integration",
                "a1111_url": "API URL",
                "a1111_enabled": "Send to A1111",
                "send_to_a1111": "Send to A1111",
                "language": "Language",
                "help": "Help",
                "exit": "Exit",
                "ready": "Ready",
                "loading_model": "Loading model...",
                "model_loaded": "Model loaded",
                "processing": "Processing...",
                "done": "Done",
                "error": "Error",
                "warning": "Warning",
                "info": "Info",
                "no_image": "No image",
                "select_image_first": "Select an image first",
                "select_folder": "Select folder",
                "processing_cancelled": "Processing cancelled",
                "processing_complete": "Processing complete",
                "save_tags": "Save tags",
                "copy_tags": "Copy tags",
                "paste": "Paste",
            }
        }
        if os.path.exists(TRANSLATIONS_FILE):
            try:
                with open(TRANSLATIONS_FILE, 'r', encoding='utf-8') as f:
                    custom = json.load(f)
                    for lang, strings in custom.items():
                        if lang in default:
                            default[lang].update(strings)
                        else:
                            default[lang] = strings
            except Exception as e:
                logging.error(f"Ошибка загрузки переводов: {e}")
        return default

    def tr(self, key: str) -> str:
        return self.current_strings.get(key, key)

    def load_model_in_thread(self):
        def load():
            self.root.after(0, lambda: self.status_var.set(self.tr("loading_model")))
            try:
                (self.session, self.all_tags,
                 self.rating_idx, self.general_idx, self.character_idx,
                 self.artist_idx, self.copyright_idx, self.other_idx) = load_model_and_tags(self.current_model_key)
                self.root.after(0, lambda: self.status_var.set(self.tr("model_loaded")))
            except Exception as e:
                logging.error(f"Ошибка загрузки модели: {e}")
                self.root.after(0, lambda: messagebox.showerror(self.tr("error"), str(e)))
        threading.Thread(target=load, daemon=True).start()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                self.general_threshold.set(cfg.get('general_threshold', 0.35))
                self.character_threshold.set(cfg.get('character_threshold', 0.85))
                self.artist_threshold.set(cfg.get('artist_threshold', 0.75))
                self.copyright_threshold.set(cfg.get('copyright_threshold', 0.75))
                self.rating_threshold.set(cfg.get('rating_threshold', 0.5))
                self.use_mcut.set(cfg.get('use_mcut', False))
                self.mcut_edge_ratio.set(cfg.get('mcut_edge_ratio', 0.9))
                self.mcut_min_ratio.set(cfg.get('mcut_min_ratio', 0.1))
                self.remove_overlap.set(cfg.get('remove_overlap', True))
                self.output_format.set(cfg.get('output_format', 'a1111'))
                self.a1111_prefix.set(cfg.get('a1111_prefix', ''))
                self.a1111_suffix.set(cfg.get('a1111_suffix', ''))
                self.group_by_category.set(cfg.get('group_by_category', False))
                self.use_blacklist.set(cfg.get('use_blacklist', False))
                self.use_whitelist.set(cfg.get('use_whitelist', False))
                self.lang.set(cfg.get('lang', 'ru'))
                self.check_updates.set(cfg.get('check_updates', True))
                self.profiling.set(cfg.get('profiling', False))
                self.a1111_url.set(cfg.get('a1111_url', 'http://127.0.0.1:7860'))
                self.a1111_enabled.set(cfg.get('a1111_enabled', False))
                self.current_model_key = cfg.get('model', DEFAULT_MODEL)

                self.blacklist = load_string_list(BLACKLIST_FILE)
                self.whitelist = load_string_list(WHITELIST_FILE)

                self.current_strings = self.translations.get(self.lang.get(), self.translations['ru'])
                logging.info("Конфигурация загружена")
            except Exception as e:
                logging.error(f"Ошибка загрузки конфига: {e}")

    def save_config(self):
        cfg = {
            'general_threshold': self.general_threshold.get(),
            'character_threshold': self.character_threshold.get(),
            'artist_threshold': self.artist_threshold.get(),
            'copyright_threshold': self.copyright_threshold.get(),
            'rating_threshold': self.rating_threshold.get(),
            'use_mcut': self.use_mcut.get(),
            'mcut_edge_ratio': self.mcut_edge_ratio.get(),
            'mcut_min_ratio': self.mcut_min_ratio.get(),
            'remove_overlap': self.remove_overlap.get(),
            'output_format': self.output_format.get(),
            'a1111_prefix': self.a1111_prefix.get(),
            'a1111_suffix': self.a1111_suffix.get(),
            'group_by_category': self.group_by_category.get(),
            'use_blacklist': self.use_blacklist.get(),
            'use_whitelist': self.use_whitelist.get(),
            'lang': self.lang.get(),
            'check_updates': self.check_updates.get(),
            'profiling': self.profiling.get(),
            'a1111_url': self.a1111_url.get(),
            'a1111_enabled': self.a1111_enabled.get(),
            'model': self.current_model_key,
        }
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            logging.info("Конфигурация сохранена")
        except Exception as e:
            logging.error(f"Ошибка сохранения конфига: {e}")

    def on_closing(self):
        self.save_config()
        self.root.destroy()

    def create_widgets(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self.tr("file"), menu=file_menu)
        file_menu.add_command(label=self.tr("open_image"), command=self.open_image)
        file_menu.add_command(label=self.tr("batch_process"), command=self.batch_process_dialog)
        file_menu.add_separator()
        file_menu.add_command(label=self.tr("exit"), command=self.on_closing)

        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self.tr("edit"), menu=edit_menu)
        edit_menu.add_command(label=self.tr("copy_tags"), command=self.copy_tags)
        edit_menu.add_command(label=self.tr("paste"), command=self.paste_tags)

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self.tr("view"), menu=view_menu)

        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self.tr("tools"), menu=tools_menu)
        tools_menu.add_command(label=self.tr("load_blacklist"), command=self.load_blacklist_dialog)
        tools_menu.add_command(label=self.tr("load_whitelist"), command=self.load_whitelist_dialog)
        tools_menu.add_separator()
        tools_menu.add_command(label=self.tr("send_to_a1111"), command=self.send_to_a1111)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self.tr("help"), menu=help_menu)
        help_menu.add_command(label=self.tr("help"), command=self.show_help)

        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text=self.tr("open_image"), command=self.open_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text=self.tr("batch_process"), command=self.batch_process_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text=self.tr("generate"), command=self.generate_tags_thread).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text=self.tr("copy"), command=self.copy_tags).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text=self.tr("save"), command=self.save_tags_dialog).pack(side=tk.LEFT, padx=2)

        self.image_label = ttk.Label(left_frame, relief=tk.SUNKEN, anchor=tk.CENTER)
        self.image_label.pack(fill=tk.BOTH, expand=True, pady=5)

        self.progress = ttk.Progressbar(left_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)

        right_frame = ttk.Frame(main_frame, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        right_frame.pack_propagate(False)

        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        settings_frame = ttk.Frame(notebook)
        notebook.add(settings_frame, text=self.tr("settings"))

        canvas = tk.Canvas(settings_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(settings_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        row = 0

        ttk.Label(scrollable_frame, text=self.tr("model")).grid(row=row, column=0, sticky=tk.W, pady=2)
        self.model_var = tk.StringVar(value=self.current_model_key)
        model_combo = ttk.Combobox(scrollable_frame, textvariable=self.model_var,
                                    values=list(MODELS.keys()), state="readonly")
        model_combo.grid(row=row, column=1, sticky=tk.W+tk.E, pady=2)
        model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        row += 1

        ttk.Label(scrollable_frame, text=self.tr("general_threshold")).grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Scale(scrollable_frame, from_=0.0, to=1.0, variable=self.general_threshold, orient=tk.HORIZONTAL).grid(row=row, column=1, sticky=tk.W+tk.E, pady=2)
        row += 1

        ttk.Label(scrollable_frame, text=self.tr("character_threshold")).grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Scale(scrollable_frame, from_=0.0, to=1.0, variable=self.character_threshold, orient=tk.HORIZONTAL).grid(row=row, column=1, sticky=tk.W+tk.E, pady=2)
        row += 1

        ttk.Label(scrollable_frame, text=self.tr("artist_threshold")).grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Scale(scrollable_frame, from_=0.0, to=1.0, variable=self.artist_threshold, orient=tk.HORIZONTAL).grid(row=row, column=1, sticky=tk.W+tk.E, pady=2)
        row += 1

        ttk.Label(scrollable_frame, text=self.tr("copyright_threshold")).grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Scale(scrollable_frame, from_=0.0, to=1.0, variable=self.copyright_threshold, orient=tk.HORIZONTAL).grid(row=row, column=1, sticky=tk.W+tk.E, pady=2)
        row += 1

        ttk.Label(scrollable_frame, text=self.tr("rating_threshold")).grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Scale(scrollable_frame, from_=0.0, to=1.0, variable=self.rating_threshold, orient=tk.HORIZONTAL).grid(row=row, column=1, sticky=tk.W+tk.E, pady=2)
        row += 1

        ttk.Checkbutton(scrollable_frame, text=self.tr("use_mcut"), variable=self.use_mcut).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1

        ttk.Label(scrollable_frame, text=self.tr("mcut_edge_ratio")).grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Scale(scrollable_frame, from_=0.0, to=1.0, variable=self.mcut_edge_ratio, orient=tk.HORIZONTAL).grid(row=row, column=1, sticky=tk.W+tk.E, pady=2)
        row += 1

        ttk.Label(scrollable_frame, text=self.tr("mcut_min_ratio")).grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Scale(scrollable_frame, from_=0.0, to=1.0, variable=self.mcut_min_ratio, orient=tk.HORIZONTAL).grid(row=row, column=1, sticky=tk.W+tk.E, pady=2)
        row += 1

        ttk.Checkbutton(scrollable_frame, text=self.tr("remove_overlap"), variable=self.remove_overlap).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1

        ttk.Label(scrollable_frame, text=self.tr("output_format")).grid(row=row, column=0, sticky=tk.W, pady=2)
        format_frame = ttk.Frame(scrollable_frame)
        format_frame.grid(row=row, column=1, sticky=tk.W, pady=2)
        ttk.Radiobutton(format_frame, text=self.tr("simple"), variable=self.output_format, value="simple").pack(anchor=tk.W)
        ttk.Radiobutton(format_frame, text=self.tr("with_probs"), variable=self.output_format, value="probs").pack(anchor=tk.W)
        ttk.Radiobutton(format_frame, text=self.tr("a1111"), variable=self.output_format, value="a1111").pack(anchor=tk.W)
        ttk.Radiobutton(format_frame, text=self.tr("grouped"), variable=self.output_format, value="grouped").pack(anchor=tk.W)
        row += 1

        ttk.Label(scrollable_frame, text=self.tr("a1111_prefix")).grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(scrollable_frame, textvariable=self.a1111_prefix).grid(row=row, column=1, sticky=tk.W+tk.E, pady=2)
        row += 1

        ttk.Label(scrollable_frame, text=self.tr("a1111_suffix")).grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(scrollable_frame, textvariable=self.a1111_suffix).grid(row=row, column=1, sticky=tk.W+tk.E, pady=2)
        row += 1

        ttk.Checkbutton(scrollable_frame, text=self.tr("group_by_category"), variable=self.group_by_category).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1

        ttk.Checkbutton(scrollable_frame, text=self.tr("use_blacklist"), variable=self.use_blacklist).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1

        ttk.Checkbutton(scrollable_frame, text=self.tr("use_whitelist"), variable=self.use_whitelist).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1

        ttk.Label(scrollable_frame, text=self.tr("a1111_integration")).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1

        ttk.Label(scrollable_frame, text=self.tr("a1111_url")).grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(scrollable_frame, textvariable=self.a1111_url).grid(row=row, column=1, sticky=tk.W+tk.E, pady=2)
        row += 1

        ttk.Checkbutton(scrollable_frame, text=self.tr("a1111_enabled"), variable=self.a1111_enabled).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1

        ttk.Label(scrollable_frame, text=self.tr("language")).grid(row=row, column=0, sticky=tk.W, pady=2)
        lang_combo = ttk.Combobox(scrollable_frame, textvariable=self.lang,
                                   values=["ru", "en"], state="readonly")
        lang_combo.grid(row=row, column=1, sticky=tk.W+tk.E, pady=2)
        lang_combo.bind('<<ComboboxSelected>>', self.on_language_change)
        row += 1

        ttk.Checkbutton(scrollable_frame, text=self.tr("check_updates"), variable=self.check_updates).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1

        ttk.Checkbutton(scrollable_frame, text=self.tr("profiling"), variable=self.profiling).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1

        output_frame = ttk.Frame(notebook)
        notebook.add(output_frame, text=self.tr("output"))

        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=20)
        self.output_text.pack(fill=tk.BOTH, expand=True)

        history_frame = ttk.Frame(notebook)
        notebook.add(history_frame, text=self.tr("history"))

        self.history_listbox = tk.Listbox(history_frame)
        self.history_listbox.pack(fill=tk.BOTH, expand=True)
        self.history_listbox.bind('<Double-Button-1>', self.on_history_select)

        ttk.Button(history_frame, text=self.tr("clear_history"), command=self.clear_history).pack(pady=5)

        self.status_var = tk.StringVar(value=self.tr("ready"))
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def on_language_change(self, event=None):
        self.current_strings = self.translations.get(self.lang.get(), self.translations['ru'])
        messagebox.showinfo(self.tr("info"), "Перезапустите приложение для смены языка.")

    def on_model_change(self, event=None):
        self.current_model_key = self.model_var.get()
        self.load_model_in_thread()

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if file_path:
            self.load_image(file_path)

    def load_image(self, path):
        self.image_path = path
        try:
            image = Image.open(path)
            image.thumbnail((500, 500), Image.Resampling.LANCZOS)
            self.current_photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.current_photo)
            self.status_var.set(f"{self.tr('loaded')}: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror(self.tr("error"), str(e))

    def on_drop(self, event):
        files = self.root.tk.splitlist(event.data)
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                self.load_image(f)
                break

    def generate_tags_thread(self):
        if self.image_path is None:
            messagebox.showwarning(self.tr("warning"), self.tr("select_image_first"))
            return
        if self.session is None:
            messagebox.showwarning(self.tr("warning"), self.tr("loading_model"))
            return
        self.progress.start()
        self.status_var.set(self.tr("processing"))
        threading.Thread(target=self.generate_tags, daemon=True).start()

    def generate_tags(self):
        try:
            if self.profiling.get():
                start_time = time.time()

            cache_key = self.image_path
            cached = self.cache.get(cache_key)
            if cached is not None:      
                probs = cached
            else:
               input_tensor = preprocess_image(self.image_path)
               input_name = self.session.get_inputs()[0].name
               output_name = self.session.get_outputs()[0].name
               outputs = self.session.run([output_name], {input_name: input_tensor})
               probs = 1 / (1 + np.exp(-outputs[0].flatten()))
               self.cache.set(cache_key, probs)

            tags_with_probs = self.filter_tags(probs)

            output_text, simple_tags = self.format_output(tags_with_probs)

            self.history.append((self.image_path, simple_tags))
            self.update_history_list()

            self.generated_simple_tags = simple_tags
            self.root.after(0, self.update_output, output_text)

            if self.profiling.get():
                elapsed = time.time() - start_time
                logging.info(f"Генерация заняла {elapsed:.2f} с")

        except Exception as e:
            logging.exception("Ошибка при генерации тегов")
            self.root.after(0, messagebox.showerror, self.tr("error"), str(e))
        finally:
            self.root.after(0, self.progress.stop)
            self.root.after(0, lambda: self.status_var.set(self.tr("done")))

    def filter_tags(self, probs):
        if self.use_mcut.get():
            gen_th = mcut_threshold(probs[self.general_idx], self.mcut_edge_ratio.get(), self.mcut_min_ratio.get())
            char_th = mcut_threshold(probs[self.character_idx], self.mcut_edge_ratio.get(), self.mcut_min_ratio.get())
            artist_th = mcut_threshold(probs[self.artist_idx], self.mcut_edge_ratio.get(), self.mcut_min_ratio.get())
            copyright_th = mcut_threshold(probs[self.copyright_idx], self.mcut_edge_ratio.get(), self.mcut_min_ratio.get())
            rating_th = mcut_threshold(probs[self.rating_idx], self.mcut_edge_ratio.get(), self.mcut_min_ratio.get())
        else:
            gen_th = self.general_threshold.get()
            char_th = self.character_threshold.get()
            artist_th = self.artist_threshold.get()
            copyright_th = self.copyright_threshold.get()
            rating_th = self.rating_threshold.get()

        selected_indices = []

        if self.rating_idx:
            best = np.argmax(probs[self.rating_idx])
            selected_indices.append(self.rating_idx[best])

        if self.general_idx:
            mask = probs[self.general_idx] >= gen_th
            selected_indices.extend(np.array(self.general_idx)[mask].tolist())

        if self.character_idx:
            mask = probs[self.character_idx] >= char_th
            selected_indices.extend(np.array(self.character_idx)[mask].tolist())

        if self.artist_idx:
            mask = probs[self.artist_idx] >= artist_th
            selected_indices.extend(np.array(self.artist_idx)[mask].tolist())

        if self.copyright_idx:
            mask = probs[self.copyright_idx] >= copyright_th
            selected_indices.extend(np.array(self.copyright_idx)[mask].tolist())

        if self.other_idx:
            mask = probs[self.other_idx] >= gen_th
            selected_indices.extend(np.array(self.other_idx)[mask].tolist())

        selected_indices.sort(key=lambda i: probs[i], reverse=True)

        tags_with_probs = [(self.all_tags[i], probs[i]) for i in selected_indices]

        if self.use_blacklist.get() and self.blacklist:
            tags_with_probs = [(t,p) for t,p in tags_with_probs if t not in self.blacklist]

        if self.use_whitelist.get() and self.whitelist:
            tags_with_probs = [(t,p) for t,p in tags_with_probs if t in self.whitelist]

        if self.remove_overlap.get():
            tags_with_probs = self.filter_overlap(tags_with_probs)

        return tags_with_probs

    def filter_overlap(self, tags_with_probs):
        tags_with_probs.sort(key=lambda x: len(x[0]), reverse=True)
        filtered = []
        seen = set()
        for tag, prob in tags_with_probs:
            if not any(tag in existing for existing in seen):
                filtered.append((tag, prob))
                seen.add(tag)
        return filtered

    def format_output(self, tags_with_probs):
        fmt = self.output_format.get()

        if fmt == "simple":
            simple = ", ".join([t for t,_ in tags_with_probs])
            return simple, simple
        elif fmt == "probs":
            lines = [f"{t}: {p:.3f}" for t,p in tags_with_probs]
            text = "\n".join(lines)
            simple = ", ".join([t for t,_ in tags_with_probs])
            return text, simple
        elif fmt == "a1111":
            prefix = self.a1111_prefix.get()
            suffix = self.a1111_suffix.get()
            simple = ", ".join([t for t,_ in tags_with_probs])
            return prefix + simple + suffix, simple
        elif fmt == "grouped":
            simple = ", ".join([t for t,_ in tags_with_probs])
            return simple, simple
        else:
            return "", ""

    def update_output(self, text):
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, text)

    def copy_tags(self):
        if hasattr(self, 'generated_simple_tags') and self.generated_simple_tags:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.generated_simple_tags)
            self.status_var.set(self.tr("copied"))
        else:
            messagebox.showinfo(self.tr("info"), self.tr("no_tags"))

    def paste_tags(self):
        try:
            text = self.root.clipboard_get()
            self.output_text.insert(tk.END, text)
        except:
            pass

    def save_tags_dialog(self):
        if not hasattr(self, 'generated_simple_tags') or not self.generated_simple_tags:
            messagebox.showwarning(self.tr("warning"), self.tr("no_tags"))
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.generated_simple_tags)
            self.status_var.set(f"{self.tr('saved')} {os.path.basename(file_path)}")

    def update_history_list(self):
        self.history_listbox.delete(0, tk.END)
        for path, tags in self.history:
            self.history_listbox.insert(tk.END, os.path.basename(path))

    def on_history_select(self, event):
        selection = self.history_listbox.curselection()
        if selection:
            idx = selection[0]
            path, tags = list(self.history)[idx]
            self.load_image(path)
            self.generated_simple_tags = tags
            self.update_output(tags)

    def clear_history(self):
        self.history.clear()
        self.update_history_list()

    def load_blacklist_dialog(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if path:
            self.blacklist = load_string_list(path)
            save_string_list(BLACKLIST_FILE, self.blacklist)
            messagebox.showinfo(self.tr("info"), f"Загружено {len(self.blacklist)} тегов")

    def load_whitelist_dialog(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if path:
            self.whitelist = load_string_list(path)
            save_string_list(WHITELIST_FILE, self.whitelist)
            messagebox.showinfo(self.tr("info"), f"Загружено {len(self.whitelist)} тегов")

    def batch_process_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title(self.tr("batch_process"))
        dialog.geometry("400x300")

        ttk.Label(dialog, text="Папка с изображениями:").pack(pady=5)
        folder_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=folder_var, width=50).pack(pady=5)
        ttk.Button(dialog, text="Обзор", command=lambda: folder_var.set(filedialog.askdirectory())).pack(pady=5)

        ttk.Label(dialog, text="Маска файлов (через запятую):").pack(pady=5)
        mask_var = tk.StringVar(value="*.jpg,*.jpeg,*.png,*.bmp,*.webp")
        ttk.Entry(dialog, textvariable=mask_var, width=50).pack(pady=5)

        ttk.Label(dialog, text="Папка для сохранения:").pack(pady=5)
        out_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=out_var, width=50).pack(pady=5)
        ttk.Button(dialog, text="Обзор", command=lambda: out_var.set(filedialog.askdirectory())).pack(pady=5)

        def start_batch():
            folder = folder_var.get()
            masks = mask_var.get().split(',')
            out_folder = out_var.get()
            if not folder or not out_folder:
                messagebox.showwarning(self.tr("warning"), "Заполните все поля")
                return
            dialog.destroy()
            self.batch_process(folder, masks, out_folder)

        ttk.Button(dialog, text="Запуск", command=start_batch).pack(pady=20)

    def batch_process(self, folder, masks, out_folder):
        files = []
        for mask in masks:
            mask = mask.strip()
            files.extend(glob.glob(os.path.join(folder, '**', mask), recursive=True))
        if not files:
            messagebox.showinfo(self.tr("info"), "Нет файлов для обработки")
            return

        def process():
            total = len(files)
            for i, f in enumerate(files):
                self.root.after(0, lambda: self.status_var.set(f"Обработка {i+1}/{total}: {os.path.basename(f)}"))
                try:
                    input_tensor = preprocess_image(f)
                    input_name = self.session.get_inputs()[0].name
                    output_name = self.session.get_outputs()[0].name
                    outputs = self.session.run([output_name], {input_name: input_tensor})
                    probs = 1 / (1 + np.exp(-outputs[0].flatten()))
                    tags_with_probs = self.filter_tags(probs)
                    _, simple_tags = self.format_output(tags_with_probs)

                    base = os.path.splitext(os.path.basename(f))[0]
                    out_path = os.path.join(out_folder, base + ".txt")
                    with open(out_path, "w", encoding="utf-8") as out_f:
                        out_f.write(simple_tags)
                except Exception as e:
                    logging.error(f"Ошибка при обработке {f}: {e}")
            self.root.after(0, lambda: messagebox.showinfo(self.tr("info"), self.tr("processing_complete")))
            self.root.after(0, lambda: self.status_var.set(self.tr("ready")))

        threading.Thread(target=process, daemon=True).start()

    def send_to_a1111(self):
        if not hasattr(self, 'generated_simple_tags') or not self.generated_simple_tags:
            messagebox.showwarning(self.tr("warning"), self.tr("no_tags"))
            return
        if not self.a1111_enabled.get():
            messagebox.showwarning(self.tr("warning"), "Интеграция A1111 отключена в настройках")
            return

        url = self.a1111_url.get().rstrip('/') + '/sdapi/v1/txt2img'
        payload = {
            "prompt": self.generated_simple_tags,
            "steps": 20,
            "cfg_scale": 7,
            "width": 512,
            "height": 512,
            "sampler_name": "Euler a",
        }
        try:
            self.status_var.set("Отправка в A1111...")
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                import base64
                from io import BytesIO
                img_data = base64.b64decode(data['images'][0])
                img = Image.open(BytesIO(img_data))
                img.show()
                self.status_var.set("Изображение получено")
            else:
                messagebox.showerror(self.tr("error"), f"Ошибка API: {response.status_code}")
        except Exception as e:
            messagebox.showerror(self.tr("error"), str(e))

    def show_help(self, event=None):
        help_text = """
        Image to Text v3.0

        Горячие клавиши:
        F1 - эта справка

        Основные функции:
        - Открыть изображение: загрузить одно изображение.
        - Пакетная обработка: обработать все изображения в папке.
        - Генерация тегов: запустить модель для текущего изображения.
        - Копировать/Сохранить: скопировать теги в буфер или сохранить в файл.

        Настройки:
        - Пороги для разных категорий (общие, персонажи, художники и т.д.)
        - MCut: автоматический подбор порога на основе распределения уверенности.
        - Удаление пересекающихся тегов: убирает дубли (cat и cat_(animal)).
        - Формат вывода: простой список, с вероятностями, для Automatic1111 (с префиксом/суффиксом), группировка по категориям.
        - Чёрный/белый список: загрузить из текстового файла (по одному тегу на строку).
        - Интеграция с Automatic1111: отправить теги в API и показать сгенерированное изображение.

        История: двойной клик по элементу истории загружает изображение снова.

        Кэширование: предобработанные тензоры и выходы модели кэшируются для ускорения повторной обработки.

        Автообновление модели: при запуске проверяется наличие новой версии модели на Hugging Face.
        """
        messagebox.showinfo(self.tr("help"), help_text)

if __name__ == "__main__":
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
        print("Для поддержки drag-and-drop установите tkinterdnd2: pip install tkinterdnd2")
    app = TaggerApp(root)
    root.mainloop()
