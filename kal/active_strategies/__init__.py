from typing import Dict, Callable

from .adv_bim import AdversarialBIMSampling
from .adv_deepfool import AdversarialDeepFoolSampling
from .entropy import EntropySampling, EntropyDropoutSampling
from .random import RandomSampling
from .strategy import Strategy
from .supervised import SupervisedSampling
from .kal import KALSampling, KALDropoutSampling, KALUncSampling, KALDropoutUncSampling
from .uncertainty import UncertaintySampling, UncertaintyDropoutSampling
from .margin import MarginSampling, MarginDropoutSampling
from .kmeans import KMeansSampling
from .kcenter import KCenterSampling
from .bald import BALDSampling

# from .least_confidence import LeastConfidenceSampling
# from .entropy_sampling import EntropySampling
# # from .least_confidence_dropout import LeastConfidenceDropout
# # # from .margin_sampling_dropout import MarginSamplingDropout
# # from .entropy_sampling_dropout import EntropySamplingDropout
# from .kmeans_sampling import KMeansSampling
# from .kcenter_greedy import KCenterGreedy
# from .bayesian_active_learning_disagreement_dropout import BALDDropout
# from .adversarial_bim import AdversarialBIM
# from .adversarial_deepfool import AdversarialDeepFool


SUPERVISED = "Supervised"
RANDOM = "Random"
KAL = "KAL"
KAL_U = "KAL_U"
KAL_D = "KAL_D"
KAL_DU = "KAL_DU"
UNCERTAINTY = "Uncertainty"
UNCERTAINTY_D = "Uncertainty_D"
MARGIN = "Margin"
MARGIN_D = "Margin_D"
KMEANS = "CoreSet_KMeans"
KCENTER = "CoreSet_KCenter"
ENTROPY = "Entropy"
ENTROPY_D = "Entropy_D"
BALD = "BALD"
ADV_DEEPFOOL = "Adv_DeepFool"
ADV_BIM = "Adv_BIM"

STRATEGIES = [
    KAL_U,
    SUPERVISED,
    BALD,
    RANDOM,
    MARGIN,
    MARGIN_D,
    UNCERTAINTY,
    UNCERTAINTY_D,
    ENTROPY,
    ENTROPY_D,
    KMEANS,
    KCENTER,
    ADV_BIM,
    ADV_DEEPFOOL,
]

SAMPLING_STRATEGIES: Dict[str, Callable[..., Strategy]] = {
    SUPERVISED: SupervisedSampling,
    RANDOM: RandomSampling,
    KAL: KALSampling,
    KAL_D: KALDropoutSampling,
    KAL_U: KALUncSampling,
    KAL_DU: KALDropoutUncSampling,
    UNCERTAINTY: UncertaintySampling,
    UNCERTAINTY_D: UncertaintyDropoutSampling,
    MARGIN: MarginSampling,
    MARGIN_D: MarginDropoutSampling,
    KMEANS: KMeansSampling,
    KCENTER: KCenterSampling,
    ENTROPY: EntropySampling,
    ENTROPY_D: EntropyDropoutSampling,
    BALD: BALDSampling,
    ADV_DEEPFOOL: AdversarialDeepFoolSampling,
    ADV_BIM: AdversarialBIMSampling
}