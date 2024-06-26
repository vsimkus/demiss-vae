from enum import Enum, auto

EPSILON=1e-5

class DISTRIBUTION(Enum):
    normal = auto()
    bern = auto()
    cont_bern = auto()
    cont_bern_prob = auto()
    cont_bern_orig = auto()
    cont_bern_orig_prob = auto()
    normal_with_eps = auto()
    stratified_mixture5_normal_with_eps = auto()
    stratified_mixture15_normal_with_eps = auto()
    stratified_mixture25_normal_with_eps = auto()
    stratified_mixture5_normal = auto()
    stratified_mixture15_normal = auto()
    stratified_mixture25_normal = auto()
    bern_m1_p1 = auto()
    stratified_mixture1_normal_with_eps = auto()
    stratified_mixture1_normal = auto()
    reparametrised_mixture5_normal_with_eps = auto()
    reparametrised_mixture15_normal_with_eps = auto()
    reparametrised_mixture25_normal_with_eps = auto()
    reparametrised_mixture5_normal = auto()
    reparametrised_mixture15_normal = auto()
    reparametrised_mixture25_normal = auto()


class PRIOR_DISTRIBUTION(Enum):
    std_normal = auto()
