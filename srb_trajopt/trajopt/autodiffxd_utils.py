from pydrake.autodiffutils import AutoDiffXd, ExtractGradient, ExtractValue
import numpy as np

def autodiff_array_equal(a, b):
    """
    # Need this because a==b returns True even if a = AutoDiffXd(1, [1, 2]), b= AutoDiffXd(2, [3, 4])
    # That's the behavior of AutoDiffXd in C++, also.
    """
    return np.array_equal(a, b) and np.array_equal(
        ExtractGradient(a), ExtractGradient(b)
    )

def extract_ad_value_and_gradient(ad):
    """
    Extracts value and gradient from an AutoDiffXd object
    """
    return ExtractValue(ad), ExtractGradient(ad)

def multiply_and_sum(mat_pair):
    "Returns dot product of two matrices"
    mat1, mat2 = mat_pair
    return mat1 @ mat2