import inspect
import os
from pathlib import Path
import imgaug.augmenters as iaa

from text_renderer.effect import *
from text_renderer.corpus import *
from text_renderer.config import (
    RenderCfg,
    NormPerspectiveTransformCfg,
    GeneratorCfg,
    FixedTextColorCfg,
    TextColorCfg,
    SimpleTextColorCfg,
    FixedPerspectiveTransformCfg,
)
from text_renderer.layout.same_line import SameLineLayout
from text_renderer.effect.curve import Curve
from text_renderer.layout import SameLineLayout, ExtraTextLineLayout
from text_renderer.layout.extra_text_line import ExtraTextLineLayout


CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
SRC_DIR = Path('/data/extended/text_dataset/text_renderer/indo4b/sources')
OUT_DIR = Path('/data/extended/text_dataset/text_renderer/indo4b/results')
# OUT_DIR = OUT_DIR / "results"

BG_DIR = SRC_DIR / "bg"
CHAR_DIR = SRC_DIR / "char"
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

norm_perspective_transform = NormPerspectiveTransformCfg(20, 20, 1.5)
fixed_perspective_transform = FixedPerspectiveTransformCfg(30, 30, 1.5)

padding_center_fx = Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True)
padding_fx = Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71])

curve_fx = Curve(p=1, period=180, amplitude=(4, 5)),
dropout_rand_fx = DropoutRand(p=1, dropout_p=(0.3, 0.5))
dropout_hor_fx = DropoutHorizontal(p=1, num_line=2, thickness=3)
dropout_ver_fx = DropoutVertical(p=1, num_line=15)
line_fx = Line(p=1, thickness=(3, 4))


extra_line_lay = ExtraTextLineLayout(bottom_prob=1.0)
same_line_lay = SameLineLayout(h_spacing=(0.9, 0.91))


NUM_IMAGE = 100

# def get_char_corpus():
#     return CharCorpus(
#         CharCorpusCfg(
#             text_paths=[TEXT_DIR / "long_text" / "100k.txt"],
#             filter_by_chars=True,
#             chars_file=CHAR_DIR / "eng.txt",
#             length=(5, 25),
#             char_spacing=(-0.3, 1.3),
#             **font_cfg
#         ),
#     )

def base_cfg(
    name: str, corpus, corpus_effects=None, layout_effects=None, layout=None, gray=True
):
    return GeneratorCfg(
        num_image=NUM_IMAGE,
        save_dir=OUT_DIR / name,
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            perspective_transform=norm_perspective_transform,
            gray=gray,
            layout_effects=layout_effects,
            layout=layout,
            corpus=corpus,
            corpus_effects=corpus_effects,
        ),
    )


def standard_icol_tcol_nopad():
    cfg = base_cfg(
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
        corpus_effects=Effects([]),
        layout_effects=Effects(Line(p=1)),
    )
    
    return cfg


def standard_icol_tcol_pad():
    cfg = base_cfg(
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
            Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71])
        ]),
        layout_effects=Effects(Line(p=1)),
    )
    
    return cfg

    
def random_dropout_icol_tcol_cpad():
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
            DropoutRand(p=0.5, dropout_p=(0.3, 0.5)),
            DropoutHorizontal(p=0.5, num_line=2, thickness=3),
            DropoutVertical(p=0.5, num_line=15)
        ]),
    )
    
    return cfg


def random_line_icol_tcol_cpad():
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
            Line(p=1, thickness=(3, 4))
        ]),
    )
    
    return cfg


def random_char_spacing_ictc():
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
        corpus_effects=Effects([]),
    )
    cfg.render_cfg.corpus.cfg.char_spacing = (-0.3,0.5)
    
    return cfg

def extra_text_line_ictc():
    cfg =  base_cfg(
        inspect.currentframe().f_code.co_name,
        layout=ExtraTextLineLayout(),
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
        corpus_effects=Effects([]),
    )
    
    return cfg


def curve_ictc():
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


def perspective_transform_ictc():
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
        corpus_effects=Effects([]),
    )
    
    cfg.render_cfg.perspective_transform = fixed_perspective_transform
    
    return cfg





def rand_data():
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
    standard_icol_tcol_pad(),
    standard_icol_tcol_nopad(),
    random_dropout_icol_tcol_cpad(),
    random_line_icol_tcol_cpad(),
    random_char_spacing_ictc(),
    extra_text_line_ictc(),
    curve_ictc(),
    perspective_transform_ictc(),
    rand_data(),
    
#    eng_word_data(),
#    same_line_data(),
#    extra_text_line_data(),
#    imgaug_emboss_example()
]
# fmt: on
