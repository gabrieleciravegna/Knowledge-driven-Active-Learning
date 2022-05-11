from typing import Dict, Callable

from .adv_bim import AdversarialBIMSampling
from .adv_deepfool import AdversarialDeepFoolSampling
from .entropy import EntropySampling, EntropyDropoutSampling
from .kal_plus import KALPlusSampling, KALPlusSamplingSVM, KALPlusSamplingTree, KALPlusSamplingLOF, \
    KALPlusUncDiversitySampling, KALPlusDiversitySampling, KALPlusUncSampling
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
KAL_DROP = "KAL_DROP"
KAL_DROP_U = "KAL_DROP_U"
KAL_DROP_D = "KAL_DROP_D"
KAL_DROP_DU = "KAL_DROP_DU"
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
    # KAL_PLUS,
    # KAL_PLUS_U,
    # KAL_PLUS_D,
    # KAL_PLUS_DU,
    # KAL_PLUS_DROP,
    # KAL_PLUS_DROP_U,
    # KAL_PLUS_DROP_D,
    # KAL_PLUS_DROP_DU,
    UNCERTAINTY,
    UNCERTAINTY_D,
    MARGIN,
    MARGIN_D,
    KMEANS,
    KCENTER,
    ENTROPY,
    ENTROPY_D,
    BALD,
    BALD2,
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

DROPOUTS = [
    KAL_DROP,
    KAL_DROP_U,
    KAL_DROP_D,
    KAL_DROP_DU,
    KAL_PLUS_DROP,
    KAL_PLUS_DROP_U,
    KAL_PLUS_DROP_D,
    KAL_PLUS_DROP_DU,
    ENTROPY_D,
    MARGIN_D,
    UNCERTAINTY_D,
]


TO_RERUN = DROPOUTS + [ENTROPY, BALD, BALD2]

KALS = [
    KAL_PLUS_DROP_DU,
    KAL_PLUS_DROP,
    KAL_PLUS_DROP_U,
    KAL_PLUS_DROP_D,
    KAL_PLUS_DU,
    KAL_PLUS,
    KAL_PLUS_U,
    KAL_PLUS_D,
    KAL,
    KAL_U,
    KAL_D,
    KAL_DU,
    KAL_DROP,
    KAL_DROP_U,
    KAL_DROP_D,
    KAL_DROP_DU,
]

SAMPLING_STRATEGIES: Dict[str, Callable[..., Strategy]] = {
    SUPERVISED: SupervisedSampling,
    RANDOM: RandomSampling,
    KAL: KALSampling,
    KAL_D: KALDiversitySampling,
    KAL_U: KALUncSampling,
    KAL_DU: KALDiversityUncSampling,
    KAL_PLUS: KALPlusSampling,
    KAL_PLUS_U: KALPlusUncSampling,
    KAL_PLUS_D: KALPlusDiversitySampling,
    KAL_PLUS_DU: KALPlusUncDiversitySampling,
    KAL_PLUS_DROP: KALPlusSampling,
    KAL_PLUS_DROP_U: KALPlusUncSampling,
    KAL_PLUS_DROP_D: KALPlusDiversitySampling,
    KAL_PLUS_DROP_DU: KALPlusUncDiversitySampling,
    KAL_PLUS_SVM: KALPlusSamplingSVM,
    KAL_PLUS_TREE: KALPlusSamplingTree,
    KAL_PLUS_LOF: KALPlusSamplingLOF,
    KAL_DROP: KALDropSampling,
    KAL_DROP_U: KALDropUncSampling,
    KAL_DROP_D: KALDropDiversitySampling,
    KAL_DROP_DU: KALDropDiversityUncSampling,
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
    ADV_BIM: AdversarialBIMSampling
}


NAME_MAPPINGS = {
    SUPERVISED: SUPERVISED,
    RANDOM: RANDOM,
    KAL_DU: "KAL",
    KAL_DROP_DU: "KAL$_D$",
    UNCERTAINTY: UNCERTAINTY,
    UNCERTAINTY_D: UNCERTAINTY+"$_D$",
    MARGIN: MARGIN,
    MARGIN_D: MARGIN+"$_D$",
    KMEANS: "CoreSet$_{KMEANS}$",
    KCENTER: "CoreSet$_{KCENTER}$",
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
    UNCERTAINTY: UNCERTAINTY,
    UNCERTAINTY_D: UNCERTAINTY + "{\\tiny $_D$}",
    MARGIN: MARGIN,
    MARGIN_D: MARGIN + "{\\tiny $_D$}",
    KMEANS: "CoreSet{\\tiny $_{KMEANS}$}",
    KCENTER: "CoreSet{\\tiny $_{KCENTER}$}",
    ENTROPY: ENTROPY,
    ENTROPY_D: ENTROPY + "{\\tiny $_D$}",
    BALD: BALD,
    BALD2: BALD+"2",
    ADV_BIM: "ADV{\\tiny $_{BIM}$}",
    ADV_DEEPFOOL: "ADV{\\tiny $_{DEEPFOOL}$}",
}