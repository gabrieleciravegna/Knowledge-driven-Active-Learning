from typing import Dict, Callable

import seaborn as sns

from .adv_bim import AdversarialBIMSampling
from .adv_deepfool import AdversarialDeepFoolSampling
from .entropy import EntropySampling, EntropyDropoutSampling
from .kal_xai import KALXAIDiversityUncSampling, KALXAIDropDiversityUncSampling
from .kal_plus import KALPlusSampling, KALPlusSamplingSVM, KALPlusSamplingTree, KALPlusSamplingLOF, \
    KALPlusUncDiversitySampling, KALPlusDiversitySampling, KALPlusUncSampling, KALPlusDropSampling, \
    KALPlusDropUncSampling, KALPlusDropDiversitySampling, KALPlusDropDiversityUncSampling
from .random import RandomSampling
from .strategy import Strategy
from .supervised import SupervisedSampling
from .kal import KALSampling, KALDiversitySampling, KALUncSampling, KALDiversityUncSampling, KALDropSampling, \
    KALDropUncSampling, KALDropDiversitySampling, KALDropDiversityUncSampling
from .uncertainty import UncertaintySampling, UncertaintyDropoutSampling
from .margin import MarginSampling, MarginDropoutSampling
from .kmeans import KMeansSampling
from .kcenter import KCenterSampling
from .bald import BALDSampling, BALDSampling2

SUPERVISED = "Supervised"
RANDOM = "Random"
KAL = "KAL"
KAL_U = "KAL_U"
KAL_D = "KAL_D"
KAL_DU = "KAL_DU"
KAL_DROP = "KAL_DROP"
KAL_DROP_U = "KAL_DROP_U"
KAL_DROP_D = "KAL_DROP_D"
KAL_DROP_DU = "KAL_DROP_DU"
KAL_PLUS = "KAL+"
KAL_PLUS_U = "KAL+_U"
KAL_PLUS_D = "KAL+_D"
KAL_PLUS_DU = "KAL+_DU"
KAL_PLUS_DROP = "KAL+_DROP"
KAL_PLUS_DROP_U = "KAL+_DROP_U"
KAL_PLUS_DROP_D = "KAL+_DROP_D"
KAL_PLUS_DROP_DU = "KAL+_DROP_DU"
KAL_PLUS_SVM = "KAL+_SVM"
KAL_PLUS_TREE = "KAL+_TREE"
KAL_PLUS_LOF = "KAL+_LOF"
KAL_STAR = "KAL_STAR"
KAL_STAR_D = "KAL_STAR_D"
KAL_STAR_U = "KAL_STAR_U"
KAL_STAR_DU = "KAL_STAR_DU"
KAL_STAR_DROP = "KAL_STAR_DROP"
KAL_STAR_DROP_D = "KAL_STAR_DROP_D"
KAL_STAR_DROP_U = "KAL_STAR_DROP_U"
KAL_STAR_DROP_DU = "KAL_STAR_DROP_DU"
KAL_LEN = "KAL_LEN"
KAL_LEN_D = "KAL_LEN_D"
KAL_LEN_U = "KAL_LEN_U"
KAL_LEN_DU = "KAL_LEN_DU"
KAL_LEN_DU_00 = "KAL_LEN_DU_00"
KAL_LEN_DU_25 = "KAL_LEN_DU_25"
KAL_LEN_DU_50 = "KAL_LEN_DU_50"
KAL_LEN_DU_75 = "KAL_LEN_DU_75"
KAL_LEN_DROP = "KAL_LEN_DROP"
KAL_LEN_DROP_D = "KAL_LEN_DROP_D"
KAL_LEN_DROP_U = "KAL_LEN_DROP_U"
KAL_LEN_DROP_DU = "KAL_LEN_DROP_DU"
UNCERTAINTY = "Uncertainty"
UNCERTAINTY_D = "Uncertainty_D"
MARGIN = "Margin"
MARGIN_D = "Margin_D"
KMEANS = "CoreSet_KMeans"
KCENTER = "CoreSet_KCenter"
ENTROPY = "Entropy"
ENTROPY_D = "Entropy_D"
BALD = "BALD"
BALD2 = "BALD2"
ADV_DEEPFOOL = "Adv_DeepFool"
ADV_BIM = "Adv_BIM"

# KAL_0 = "KAL_00"
KAL_25 = "KAL_25"
KAL_50 = "KAL_50"
KAL_75 = "KAL_75"
KAL_DU_00 = "KAL_DU_00"
KAL_DU_25 = "KAL_DU_25"
KAL_DU_50 = "KAL_DU_50"
KAL_DU_75 = "KAL_DU_75"

STRATEGIES = [
    SUPERVISED,
    RANDOM,
    # KAL,
    # KAL_U,
    # KAL_D,
    KAL_DU,
    # KAL_DROP,
    # KAL_DROP_U,
    # KAL_DROP_D,
    KAL_DROP_DU,
    # KAL_LEN_DU,
    # KAL_LEN_DROP_DU,
    # KAL_PLUS,
    # KAL_PLUS_U,
    # KAL_PLUS_D,
    # KAL_PLUS_DU,
    # KAL_PLUS_DROP,
    # KAL_PLUS_DROP_U,
    # KAL_PLUS_DROP_D,
    # KAL_PLUS_DROP_DU,
    # KAL_STAR_DU,
    # KAL_STAR_DROP_DU,
    UNCERTAINTY,
    UNCERTAINTY_D,
    MARGIN,
    MARGIN_D,
    KMEANS,
    KCENTER,
    ENTROPY,
    ENTROPY_D,
    BALD,
    # BALD2,
    ADV_BIM,
    ADV_DEEPFOOL,
]

FAST_STRATEGIES = [
    KAL_DU,
    SUPERVISED,
    RANDOM,
    MARGIN,
    UNCERTAINTY,
    ENTROPY,
]

REGRESSION_STRATEGIES = [
    SUPERVISED,
    RANDOM,
    KAL_DU,
    KAL_DROP_DU,
    KMEANS,
    KCENTER,
]

DROPOUTS = [
    KAL_DROP,
    KAL_DROP_U,
    KAL_DROP_D,
    KAL_DROP_DU,
    KAL_LEN_DROP,
    KAL_LEN_DROP_U,
    KAL_LEN_DROP_D,
    KAL_LEN_DROP_DU,
    KAL_PLUS_DROP,
    KAL_PLUS_DROP_U,
    KAL_PLUS_DROP_D,
    KAL_PLUS_DROP_DU,
    KAL_STAR_DROP,
    KAL_STAR_DROP_U,
    KAL_STAR_DROP_D,
    KAL_STAR_DROP_DU,
    ENTROPY_D,
    MARGIN_D,
    UNCERTAINTY_D,
]

KALS = [
    KAL,
    KAL_D,
    KAL_U,
    KAL_DU,
    KAL_DROP,
    KAL_DROP_D,
    KAL_DROP_U,
    KAL_DROP_DU,
    KAL_PLUS,
    KAL_PLUS_U,
    KAL_PLUS_D,
    KAL_PLUS_DU,
    KAL_PLUS_DROP,
    KAL_PLUS_DROP_U,
    KAL_PLUS_DROP_D,
    KAL_PLUS_DROP_DU,
    KAL_25,
    KAL_50,
    KAL_75,
]

KAL_PARTIAL = [
    # KAL_0,
    KAL_25,
    KAL_50,
    KAL_75,
    KAL,
    KAL_DU_00,
    KAL_DU_25,
    KAL_DU_50,
    KAL_DU_75,
    KAL_DU
]

KAL_STARS = [
    KAL_STAR_DU,
    KAL_STAR_DROP_DU
]

KAL_LENS = [
    KAL_LEN_DU,
    KAL_LEN_DU_00,
    KAL_LEN_DU_25,
    KAL_LEN_DU_50,
    KAL_LEN_DU_75
]

SAMPLING_STRATEGIES: Dict[str, Callable[..., Strategy]] = {
    SUPERVISED: SupervisedSampling,
    RANDOM: RandomSampling,
    KAL: KALSampling,
    KAL_D: KALDiversitySampling,
    KAL_U: KALUncSampling,
    KAL_DU: KALDiversityUncSampling,
    KAL_DROP: KALDropSampling,
    KAL_DROP_U: KALDropUncSampling,
    KAL_DROP_D: KALDropDiversitySampling,
    KAL_DROP_DU: KALDropDiversityUncSampling,
    KAL_STAR_DU: KALDiversityUncSampling,
    KAL_STAR_DROP_DU: KALDropDiversityUncSampling,
    KAL_LEN_DU: KALXAIDiversityUncSampling,
    KAL_LEN_DROP_DU: KALXAIDropDiversityUncSampling,
    KAL_PLUS: KALPlusSampling,
    KAL_PLUS_U: KALPlusUncSampling,
    KAL_PLUS_D: KALPlusDiversitySampling,
    KAL_PLUS_DU: KALPlusUncDiversitySampling,
    KAL_PLUS_DROP: KALPlusDropSampling,
    KAL_PLUS_DROP_U: KALPlusDropUncSampling,
    KAL_PLUS_DROP_D: KALPlusDropDiversitySampling,
    KAL_PLUS_DROP_DU: KALPlusDropDiversityUncSampling,
    KAL_PLUS_SVM: KALPlusSamplingSVM,
    KAL_PLUS_TREE: KALPlusSamplingTree,
    KAL_PLUS_LOF: KALPlusSamplingLOF,
    UNCERTAINTY: UncertaintySampling,
    UNCERTAINTY_D: UncertaintyDropoutSampling,
    MARGIN: MarginSampling,
    MARGIN_D: MarginDropoutSampling,
    KMEANS: KMeansSampling,
    KCENTER: KCenterSampling,
    ENTROPY: EntropySampling,
    ENTROPY_D: EntropyDropoutSampling,
    BALD: BALDSampling,
    BALD2: BALDSampling2,
    ADV_DEEPFOOL: AdversarialDeepFoolSampling,
    ADV_BIM: AdversarialBIMSampling,
    # KAL_0: KALSampling,
    KAL_25: KALSampling,
    KAL_50: KALSampling,
    KAL_75: KALSampling,
    KAL_DU_00: KALDiversityUncSampling,
    KAL_DU_25: KALDiversityUncSampling,
    KAL_DU_50: KALDiversityUncSampling,
    KAL_DU_75: KALDiversityUncSampling,
    KAL_LEN_DU_00: KALDiversityUncSampling,
    KAL_LEN_DU_25: KALDiversityUncSampling,
    KAL_LEN_DU_50: KALDiversityUncSampling,
    KAL_LEN_DU_75: KALDiversityUncSampling
}

NAME_MAPPINGS_ABLATION_STUDY = {
    KAL: "KAL",
    KAL_25: "KAL 25\%",
    KAL_50: "KAL 50\%",
    KAL_75: "KAL 75\%",
    KAL_D: "KAL Div",
    KAL_U: "KAL Unc",
    KAL_DU: "KAL Div Unc",
    KAL_DROP: "KAL$_D$",
    KAL_DROP_D: "KAL$_D$ Div",
    KAL_DROP_U: "KAL$_D$ Unc",
    KAL_DROP_DU: "KAL$_D$ Div Unc",
    KAL_PLUS: "KAL$^+$",
    KAL_PLUS_D: "KAL$^+$ Div",
    KAL_PLUS_U: "KAL$^+$ Unc",
    KAL_PLUS_DU: "KAL$^+$ Div Unc",
    KAL_PLUS_DROP: "KAL$_D^+$",
    KAL_PLUS_DROP_D: "KAL$_D^+$ Div",
    KAL_PLUS_DROP_U: "KAL$_D^+$ Unc",
    KAL_PLUS_DROP_DU: "KAL$_D^+$ Div Unc",
}

NAME_MAPPINGS = {
    SUPERVISED: "SupLoss",
    RANDOM: RANDOM,
    KAL: "KAL-",
    # KAL_00: "KAL-00\%",
    KAL_25: "KAL- 25\%",
    KAL_50: "KAL- 50\%",
    KAL_75: "KAL- 75\%",
    KAL_DU: "KAL",
    KAL_DU_00: "KAL 00\%",
    KAL_DU_25: "KAL 25\%",
    KAL_DU_50: "KAL 50\%",
    KAL_DU_75: "KAL 75\%",
    KAL_DROP_DU: "KAL$_D$",
    KAL_PLUS_DU: "KAL$^+$",
    KAL_PLUS_DROP_DU: "KAL$_D^+$",
    KAL_STAR_DU: "KAL$^*$",
    KAL_STAR_DROP_DU: "KAL$_D^*$",
    UNCERTAINTY: "LeastConf",
    UNCERTAINTY_D: "LeastConf"+"$_D$",
    MARGIN: MARGIN,
    MARGIN_D: MARGIN+"$_D$",
    KMEANS: "KMEANS",
    KCENTER: "KCENTER",
    ENTROPY: ENTROPY,
    ENTROPY_D: ENTROPY+"$_D$",
    BALD: BALD,
    BALD2: BALD+"2",
    ADV_BIM: "ADV$_{BIM}$",
    ADV_DEEPFOOL: "ADV$_{DEEPFOOL}$",
}


NAME_MAPPINGS_LATEX = {
    SUPERVISED: SUPERVISED,
    RANDOM: RANDOM,
    KAL_DU: KAL,
    KAL_DROP_DU: KAL + "{\\tiny $_D$}",
    KAL_PLUS_DU: KAL + "$^+$",
    KAL_PLUS_DROP_DU: KAL + "{\\tiny $_D$}^+",
    KAL_STAR_DU: "KAL$^*$",
    KAL_STAR_DROP_DU: KAL + "{\\tiny %_D}^*$",
    UNCERTAINTY: UNCERTAINTY,
    UNCERTAINTY_D: UNCERTAINTY + "{\\tiny $_D$}",
    MARGIN: MARGIN,
    MARGIN_D: MARGIN + "{\\tiny $_D$}",
    KMEANS: "KMEANS",
    KCENTER: "KCENTER",
    ENTROPY: ENTROPY,
    ENTROPY_D: ENTROPY + "{\\tiny $_D$}",
    BALD: BALD,
    BALD2: BALD+"2",
    ADV_BIM: "ADV{\\tiny $_{BIM}$}",
    ADV_DEEPFOOL: "ADV{\\tiny $_{DEEPFOOL}$}",
}


colors = sns.color_palette()
color_mappings = {
    'Adv_BIM': colors[5],
    'Adv_DeepFool': colors[5],
    'BALD': colors[1],
    'CoreSet_KCenter': colors[2],
    'CoreSet_KMeans': colors[3],
    'Entropy': colors[4],
    'Entropy_D': colors[4],
    'KAL': colors[9],
    'KAL_DU': colors[9],
    'KAL_DROP_DU': colors[9],
    'Margin': colors[6],
    'Margin_D': colors[6],
    'Random': colors[7],
    'Supervised': colors[8],
    'SupLoss': colors[8],
    'Uncertainty': colors[0],
    'Uncertainty_D': colors[0],
}