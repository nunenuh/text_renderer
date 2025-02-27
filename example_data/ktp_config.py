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
)
from text_renderer.layout.same_line import SameLineLayout
from text_renderer.layout.extra_text_line import ExtraTextLineLayout


CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
OUT_DIR = Path('/data/extended/text_dataset/text_renderer/testing/')
OUT_DIR = OUT_DIR / "results"
DATA_DIR = CURRENT_DIR / "ktp" / "sources"
BG_DIR = DATA_DIR / "bg"
CHAR_DIR = DATA_DIR / "char"
FONT_DIR = DATA_DIR / "font" 
TEXT_DIR = DATA_DIR / "text"

print(DATA_DIR)

font_cfg = dict(
    font_dir=FONT_DIR / "font_files",
    font_list_file=FONT_DIR / "font_list.txt",
    font_size=(21, 34),
)

perspective_transform = NormPerspectiveTransformCfg(20, 20, 1.5)


def get_char_corpus():
    return CharCorpus(
        CharCorpusCfg(
            text_paths=[TEXT_DIR / "long_text" / "100k.txt"],
            filter_by_chars=True,
            chars_file=CHAR_DIR / "eng.txt",
            length=(5, 25),
            char_spacing=(-0.3, 1.3),
            **font_cfg
        ),
    )

def base_cfg(
    name: str, corpus, corpus_effects=None, layout_effects=None, layout=None, gray=True
):
    return GeneratorCfg(
        num_image=100,
        save_dir=OUT_DIR / name,
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            perspective_transform=perspective_transform,
            gray=gray,
            layout_effects=layout_effects,
            layout=layout,
            corpus=corpus,
            corpus_effects=corpus_effects,
        ),
    )


# def chn_data():
#     return base_cfg(
#         inspect.currentframe().f_code.co_name,
#         corpus=get_char_corpus(),
#         corpus_effects=Effects([Line(0.5), OneOf([DropoutRand(), DropoutVertical()])]),
#     )


def enum_data():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        layout=SameLineLayout(),
        gray=False,
        corpus=EnumCorpus(
            EnumCorpusCfg(
                text_paths=[
                    TEXT_DIR / "enum" / "wilayah_enum.txt", 
#                    TEXT_DIR / "enum" / "wilayah_text.txt", 
                    TEXT_DIR / "enum" / "nama_enum.txt", 
#                    TEXT_DIR / "enum" / "nama_text.txt", 
                    TEXT_DIR / "enum" / "alamat_enum.txt", 
#                    TEXT_DIR / "enum" / "alamat_text.txt", 
                    TEXT_DIR / "enum" / "pekerjaan_enum.txt", 
#                    TEXT_DIR / "enum" / "pekerjaan_text.txt", 
                    TEXT_DIR / "enum" / "perkawinan_enum.txt", 
#                    TEXT_DIR / "enum" / "perkawinan_text.txt", 
#                    TEXT_DIR / "enum" / "agama_text.txt", 
		     TEXT_DIR / "enum" / "nik_enum_10000r.txt", 
		    
                ],
                filter_by_chars=True,
                chars_file=CHAR_DIR / "eng.txt",
                **font_cfg
            ),
        ),
        corpus_effects=Effects([
        	Padding(),
        	Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
               ImgAugEffect(aug=iaa.Emboss(alpha=(0.9, 1.0), strength=(1.5, 1.6))),
               Line(p=1, thickness=(3, 4)),
        ]),
        layout_effects=Effects(Line(p=1)),
        

    )


def rand_data():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=RandCorpus(
            RandCorpusCfg(chars_file=CHAR_DIR / "eng.txt", **font_cfg),
        ),
    )


def eng_word_data():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=WordCorpus(
            WordCorpusCfg(
                text_paths=[TEXT_DIR / "long_text" / "100k.txt"],
                filter_by_chars=True,
                chars_file=CHAR_DIR / "eng.txt",
                **font_cfg
            ),
        ),
    )


def same_line_data():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        layout=SameLineLayout(),
        gray=False,
        corpus=[
            EnumCorpus(
                EnumCorpusCfg(
                    text_paths=[
                        TEXT_DIR / "enum" / "wilayah_enum.txt", 
                        TEXT_DIR / "enum" / "wilayah_text.txt", 
                        TEXT_DIR / "enum" / "nama_enum.txt", 
                        TEXT_DIR / "enum" / "nama_text.txt", 
                        TEXT_DIR / "enum" / "alamat_enum.txt", 
                        TEXT_DIR / "enum" / "alamat_text.txt", 
                        TEXT_DIR / "enum" / "pekerjaan_enum.txt", 
                        TEXT_DIR / "enum" / "pekerjaan_text.txt", 
                        TEXT_DIR / "enum" / "perkawinan_enum.txt", 
                        TEXT_DIR / "enum" / "perkawinan_text.txt", 
                        TEXT_DIR / "enum" / "agama_text.txt", 
                    ],
                    filter_by_chars=True,
                    chars_file=CHAR_DIR / "eng.txt",
                    **font_cfg
                ),
            ),
            CharCorpus(
                CharCorpusCfg(
                    text_paths=[
                        TEXT_DIR / "long_text" / "100k.txt"
                    ],
                    filter_by_chars=True,
                    chars_file=CHAR_DIR / "eng.txt",
                    length=(5, 10),
                    font_dir=font_cfg["font_dir"],
                    font_list_file=font_cfg["font_list_file"],
                    font_size=(30, 35),
                ),
            ),
        ],
        corpus_effects=[Effects([Padding(), DropoutRand()]), NoEffects()],
        layout_effects=Effects(Line(p=1)),
    )


def extra_text_line_data():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        layout=ExtraTextLineLayout(),
        corpus=[
            CharCorpus(
                CharCorpusCfg(
                    text_paths=[
                        # TEXT_DIR / "chn_text.txt",
                        TEXT_DIR / "long_text" / "100k.txt"
                    ],
                    filter_by_chars=True,
                    chars_file=CHAR_DIR / "eng.txt",
                    length=(9, 10),
                    font_dir=font_cfg["font_dir"],
                    font_list_file=font_cfg["font_list_file"],
                    font_size=(30, 35),
                ),
            ),
            CharCorpus(
                CharCorpusCfg(
                    text_paths=[
                        # TEXT_DIR / "chn_text.txt",
                        TEXT_DIR / "long_text" / "100k.txt"
                    ],
                    filter_by_chars=True,
                    chars_file=CHAR_DIR / "eng.txt",
                    length=(9, 10),
                    font_dir=font_cfg["font_dir"],
                    font_list_file=font_cfg["font_list_file"],
                    font_size=(30, 35),
                ),
            ),
        ],
        corpus_effects=[Effects([Padding()]), NoEffects()],
        layout_effects=Effects(Line(p=1)),
    )


def imgaug_emboss_example():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_corpus(),
        corpus_effects=Effects(
            [
                Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
                ImgAugEffect(aug=iaa.Emboss(alpha=(0.9, 1.0), strength=(1.5, 1.6))),
            ]
        ),
    )


# fmt: off
# The configuration file must have a configs variable
configs = [
    enum_data(),
#    rand_data(),
#    eng_word_data(),
#    same_line_data(),
#    extra_text_line_data(),
#    imgaug_emboss_example()
]
# fmt: on

