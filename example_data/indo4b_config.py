import inspect
import os
from pathlib import Path
from text_renderer.effect.padding import RandomCenterPadding
import imgaug.augmenters as iaa

from text_renderer.effect import *
from text_renderer.corpus import *
from text_renderer.config import (
    RenderCfg,
    NormPerspectiveTransformCfg,
    FixedPerspectiveTransformCfg,
    UniformPerspectiveTransformCfg,
    GeneratorCfg,
    SimpleTextColorCfg,
)
from text_renderer.layout.same_line import SameLineLayout
from text_renderer.effect.curve import Curve
from text_renderer.layout import SameLineLayout, ExtraTextLineLayout
from text_renderer.layout.extra_text_line import ExtraTextLineLayout

import random

CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
SRC_DIR = Path('/data/extended/text_dataset/text_renderer/indo4b/sources')
OUT_DIR = Path('/data/extended/text_dataset/text_renderer/indo4b/results')
# OUT_DIR = OUT_DIR / "results"

BG_DIR = SRC_DIR / "bg" / "general"
CHAR_DIR = SRC_DIR / "char"
CHAR_FILE = CHAR_DIR / "eng.txt"

FONT_DIR = SRC_DIR / "font" 
TEXT_DIR = SRC_DIR / "text"
TEXT_FILES = sorted(list(Path(TEXT_DIR).glob("*.txt")))
print(TEXT_FILES)

print(SRC_DIR)

font_cfg = dict(
    font_dir=FONT_DIR / "font_files",
    font_list_file=FONT_DIR / "font_list.txt",
    font_size=(21, 34),
)


NUM_IMAGE = 100

def base_cfg(
    name: str, corpus, corpus_effects=None, layout_effects=None, layout=None, gray=True
):
    return GeneratorCfg(
        num_image=NUM_IMAGE,
        save_dir=OUT_DIR / name,
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            perspective_transform=NormPerspectiveTransformCfg(20, 20, 1.5),
            gray=gray,
            layout_effects=layout_effects,
            layout=layout,
            corpus=corpus,
            corpus_effects=corpus_effects,
        ),
    )


def standard():
    cfg = base_cfg(
        inspect.currentframe().f_code.co_name,
        layout=SameLineLayout(),
        gray=False,
        corpus=EnumCorpus(
            EnumCorpusCfg(
                text_paths=TEXT_FILES,
                filter_by_chars=True,
                text_color_cfg=SimpleTextColorCfg(),
                chars_file=CHAR_FILE,
                **font_cfg
            ),
        ),
        corpus_effects=Effects([
            RandomCenterPadding(p=0.5, center_prob=0.5, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71],)
        ]),
    )
    
    return cfg

def compact_spacing():
    cfg =  base_cfg(
        inspect.currentframe().f_code.co_name,
        layout=SameLineLayout(),
        gray=False,
        corpus=EnumCorpus(
            EnumCorpusCfg(
                text_paths=TEXT_FILES,
                filter_by_chars=True,
                text_color_cfg=SimpleTextColorCfg(),
                chars_file=CHAR_DIR / "eng.txt",
                **font_cfg
            ),
        ),
        corpus_effects=Effects([
            RandomCenterPadding(p=0.5, center_prob=0.5, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71]),
        ]),
    )
    cfg.render_cfg.corpus.cfg.char_spacing = -0.3
    
    return cfg

def large_spacing():
    cfg =  base_cfg(
        inspect.currentframe().f_code.co_name,
        layout=SameLineLayout(),
        gray=False,
        corpus=EnumCorpus(
            EnumCorpusCfg(
                text_paths=TEXT_FILES,
                filter_by_chars=True,
                text_color_cfg=SimpleTextColorCfg(),
                chars_file=CHAR_DIR / "eng.txt",
                **font_cfg
            ),
        ),
        corpus_effects=Effects([
            RandomCenterPadding(p=0.5, center_prob=0.5, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71]),
        ]),
    )
    cfg.render_cfg.corpus.cfg.char_spacing = 0.5
    
    return cfg


def curve():
    cfg =  base_cfg(
        inspect.currentframe().f_code.co_name,
        layout=SameLineLayout(),
        gray=False,
        corpus=EnumCorpus(
            EnumCorpusCfg(
                text_paths=TEXT_FILES,
                filter_by_chars=True,
                text_color_cfg=SimpleTextColorCfg(),
                chars_file=CHAR_DIR / "eng.txt",
                **font_cfg
            ),
        ),
        corpus_effects=Effects([
            Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
            Curve(p=1, period=180, amplitude=(4, 5)),
        ]),
    )
    
    return cfg
    
def random_dropout():
    cfg =  base_cfg(
        inspect.currentframe().f_code.co_name,
        layout=SameLineLayout(),
        gray=False,
        corpus=EnumCorpus(
            EnumCorpusCfg(
                text_paths=TEXT_FILES,
                filter_by_chars=True,
                text_color_cfg=SimpleTextColorCfg(),
                chars_file=CHAR_DIR / "eng.txt",
                **font_cfg
            ),
        ),
        corpus_effects=Effects([
            RandomCenterPadding(p=0.5, center_prob=0.5, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71]),
            random.choice([
                DropoutRand(p=0.5, dropout_p=(0.3, 0.5)),
                DropoutHorizontal(p=0.5, num_line=2, thickness=3),
                DropoutVertical(p=0.5, num_line=15)
            ])
        ]),
    )
    
    return cfg

def random_line():
    cfg =  base_cfg(
        inspect.currentframe().f_code.co_name,
        layout=SameLineLayout(),
        gray=False,
        corpus=EnumCorpus(
            EnumCorpusCfg(
                text_paths=TEXT_FILES,
                filter_by_chars=True,
                text_color_cfg=SimpleTextColorCfg(),
                chars_file=CHAR_DIR / "eng.txt",
                **font_cfg
            ),
        ),
        corpus_effects=Effects([
            RandomCenterPadding(p=0.5, center_prob=0.5, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71]),
            Line(p=1, thickness=(3, 4))
        ]),
    )
    
    return cfg


def extra_text():
    cfg =  base_cfg(
        inspect.currentframe().f_code.co_name,
        layout=ExtraTextLineLayout(bottom_prob=1.0),
        gray=False,
        corpus=[
            EnumCorpus(
                EnumCorpusCfg(
                    text_paths=TEXT_FILES,
                    filter_by_chars=True,
                    text_color_cfg=SimpleTextColorCfg(),
                    chars_file=CHAR_DIR / "eng.txt",
                    **font_cfg
                )
            ),
            EnumCorpus(
                EnumCorpusCfg(
                    text_paths=TEXT_FILES,
                    filter_by_chars=True,
                    text_color_cfg=SimpleTextColorCfg(),
                    chars_file=CHAR_DIR / "eng.txt",
                    **font_cfg
                )
            ),
        ],
        corpus_effects=Effects([
            RandomCenterPadding(p=0.5, center_prob=0.5, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71]),
        ]),
        layout_effects=Effects(Line(p=1)),
    )
    
    return cfg

def perspective_transform():
    cfg =  base_cfg(
        inspect.currentframe().f_code.co_name,
        layout=SameLineLayout(),
        gray=False,
        corpus=EnumCorpus(
            EnumCorpusCfg(
                text_paths=TEXT_FILES,
                filter_by_chars=True,
                text_color_cfg=SimpleTextColorCfg(),
                chars_file=CHAR_DIR / "eng.txt",
                **font_cfg
            ),
        ),
        corpus_effects=Effects([
            RandomCenterPadding(p=0.5, center_prob=0.5, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71]),
        ]),
    )
    
    cfg.render_cfg.perspective_transform = UniformPerspectiveTransformCfg(30, 30, 1.5)
    
    return cfg


def random_character():
    cfg = base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=RandCorpus(
            RandCorpusCfg(chars_file=CHAR_DIR / "eng.txt", **font_cfg),
        ),
    )
    return cfg


# fmt: off
# The configuration file must have a configs variable
configs = [
    standard(),
    compact_spacing(),
    large_spacing(),
    curve(),
    random_dropout(),
    random_line(),
    extra_text(),
    perspective_transform(),
    random_character(),
]
# fmt: on
