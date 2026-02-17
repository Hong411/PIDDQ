# -*- coding: utf-8 -*-
"""
Created on Wed May 21 11:02:12 2025

@author: Bo
"""

"""
Performance metrics from external sources for comparison studies.
All metrics are stored as nested dictionaries with consistent structure.
"""

# external data
AGG_PERFORMANCE = {
    "Alkane": {
        "2-Class": {"Accuracy": 0.917, "Precision": 0.940, "Recall": 0.940, "Specificity": 0.865, "F1": 0.940},
        "m-Label": {"Accuracy": 0.939, "Precision": 0.962, "Recall": 0.966, "Specificity": 0.792, "F1": 0.964}
    },
    "Alcohol": {
        "2-Class": {"Accuracy": 0.973, "Precision": 0.950, "Recall": 0.952, "Specificity": 0.981, "F1": 0.951},
        "m-Label": {"Accuracy": 0.975, "Precision": 0.956, "Recall": 0.951, "Specificity": 0.984, "F1": 0.953}
    },
    "Alkene": {
        "2-Class": {"Accuracy": 0.961, "Precision": 0.916, "Recall": 0.776, "Specificity": 0.989, "F1": 0.840},
        "m-Label": {"Accuracy": 0.954, "Precision": 0.854, "Recall": 0.791, "Specificity": 0.980, "F1": 0.820}
    },
    "Alkyne": {
        "2-Class": {"Accuracy": 0.996, "Precision": 0.949, "Recall": 0.875, "Specificity": 0.999, "F1": 0.910},
        "m-Label": {"Accuracy": 0.993, "Precision": 0.934, "Recall": 0.810, "Specificity": 0.999, "F1": 0.866}
    },
    "Amide": {
        "2-Class": {"Accuracy": 0.980, "Precision": 0.715, "Recall": 0.590, "Specificity": 0.995, "F1": 0.892},
        "m-Label": {"Accuracy": 0.989, "Precision": 0.776, "Recall": 0.647, "Specificity": 0.996, "F1": 0.702}
    },
    "Amine": {
        "2-Class": {"Accuracy": 0.980, "Precision": 0.915, "Recall": 0.871, "Specificity": 0.992, "F1": 0.892},
        "m-Label": {"Accuracy": 0.966, "Precision": 0.910, "Recall": 0.885, "Specificity": 0.982, "F1": 0.897}
    },
    "Aromatic": {
        "2-Class": {"Accuracy": 0.977, "Precision": 0.979, "Recall": 0.981, "Specificity": 0.971, "F1": 0.980},
        "m-Label": {"Accuracy": 0.974, "Precision": 0.978, "Recall": 0.976, "Specificity": 0.970, "F1": 0.977}
    },
    "Carboxylic Acid": {
        "2-Class": {"Accuracy": 0.990, "Precision": 0.947, "Recall": 0.910, "Specificity": 0.996, "F1": 0.928},
        "m-Label": {"Accuracy": 0.989, "Precision": 0.923, "Recall": 0.928, "Specificity": 0.994, "F1": 0.925}
    },
    "Ester": {
        "2-Class": {"Accuracy": 0.986, "Precision": 0.949, "Recall": 0.928, "Specificity": 0.994, "F1": 0.938},
        "m-Label": {"Accuracy": 0.983, "Precision": 0.933, "Recall": 0.913, "Specificity": 0.992, "F1": 0.922}
    },
    "Ether": {
        "2-Class": {"Accuracy": 0.963, "Precision": 0.938, "Recall": 0.913, "Specificity": 0.980, "F1": 0.925},
        "m-Label": {"Accuracy": 0.963, "Precision": 0.879, "Recall": 0.862, "Specificity": 0.980, "F1": 0.869}
    },
    "Aldehyde": {
        "2-Class": {"Accuracy": 0.998, "Precision": 0.976, "Recall": 0.941, "Specificity": 0.999, "F1": 0.958},
        "m-Label": {"Accuracy": 0.997, "Precision": 0.973, "Recall": 0.900, "Specificity": 0.999, "F1": 0.934}
    },
    "Ketone": {
        "2-Class": {"Accuracy": 0.978, "Precision": 0.900, "Recall": 0.851, "Specificity": 0.990, "F1": 0.875},
        "m-Label": {"Accuracy": 0.978, "Precision": 0.893, "Recall": 0.868, "Specificity": 0.990, "F1": 0.879}
    },
    "Nitrile": {
        "2-Class": {"Accuracy": 0.984, "Precision": 0.899, "Recall": 0.727, "Specificity": 0.996, "F1": 0.800},
        "m-Label": {"Accuracy": 0.979, "Precision": 0.824, "Recall": 0.674, "Specificity": 0.994, "F1": 0.740}
    },
    "Nitro": {
        "2-Class": {"Accuracy": 0.995, "Precision": 0.962, "Recall": 0.941, "Specificity": 0.998, "F1": 0.951},
        "m-Label": {"Accuracy": 0.994, "Precision": 0.952, "Recall": 0.934, "Specificity": 0.997, "F1": 0.943}
    }
}

MLP_PERFORMANCE = {
    "Alkane": {
        "Training set F1": 0.966563,
        "Validation set F1": 0.932969
    },
    "Alcohol": {
        "Training set F1": 0.981538,
        "Validation set F1": 0.957765
    },
    "Alkene": {
        "Training set F1": 0.898341,
        "Validation set F1": 0.823709
    },
    "Alkyne": {
        "Training set F1": 0.946598,
        "Validation set F1": 0.847545
    },
    "Amide": {
        "Training set F1": 0.783791,
        "Validation set F1": 0.620740
    },
    "Amine": {
        "Training set F1": 0.948083,
        "Validation set F1": 0.877436
    },
    "Aromatic": {
        "Training set F1": 0.991503,
        "Validation set F1": 0.976025
    },
    "Carboxylic Acid": {
        "Training set F1": 0.974353,
        "Validation set F1": 0.944752
    },
    "Ester": {
        "Training set F1": 0.978914,
        "Validation set F1": 0.933366
    },
    "Ether": {
        "Training set F1": 0.977310,
        "Validation set F1": 0.935875
    },
    "Aldehyde": {
        "Training set F1": 0.982074,
        "Validation set F1": 0.927797
    },
    "Ketone": {
        "Training set F1": 0.952114,
        "Validation set F1": 0.882585
    },
    "Nitrile": {
        "Training set F1": 0.739183,
        "Validation set F1": 0.525128
    },
    "Nitro": {
        "Training set F1": 0.986419,
        "Validation set F1": 0.953173
    }
}

FCG_PERFORMANCE = {
    "Alcohol": {
        "Accuracy": (250 + 664) / (250 + 8 + 11 + 664),  # 0.9749
        "Precision": 0.957854406,
        "Recall": 0.968992248,
        "F1": 0.963391137
    },
    "Aldehyde": {
        "Accuracy": (14 + 918) / (14 + 0 + 1 + 918),  # 0.9989
        "Precision": 0.933333333,
        "Recall": 1.0,
        "F1": 0.965517241
    },
    "Alkane": {
        "Accuracy": (627 + 234) / (627 + 33 + 39 + 234),  # 0.9225
        "Precision": 0.941441441,
        "Recall": 0.95,
        "F1": 0.945701357
    },
    "Alkene": {
        "Accuracy": (95 + 802) / (95 + 18 + 18 + 802),  # 0.9615
        "Precision": 0.840707965,
        "Recall": 0.840707965,
        "F1": 0.840707965
    },
    "Alkyne": {
        "Accuracy": (27 + 900) / (27 + 2 + 4 + 900),  # 0.9936
        "Precision": 0.870967742,
        "Recall": 0.931034483,
        "F1": 0.9
    },
    "Amide": {
        "Accuracy": (13 + 907) / (13 + 7 + 6 + 907),  # 0.9860
        "Precision": 0.684210526,
        "Recall": 0.65,
        "F1": 0.666666667
    },
    "Amine": {
        "Accuracy": (88 + 825) / (88 + 10 + 10 + 825),  # 0.9786
        "Precision": 0.897959184,
        "Recall": 0.897959184,
        "F1": 0.897959184
    },
    "Aromatic": {
        "Accuracy": (516 + 397) / (516 + 12 + 8 + 397),  # 0.9786
        "Precision": 0.984732824,
        "Recall": 0.977272727,
        "F1": 0.980988593
    },
    "Carboxylic Acid": {
        "Accuracy": (63 + 861) / (63 + 3 + 6 + 861),  # 0.9904
        "Precision": 0.913043478,
        "Recall": 0.954545455,
        "F1": 0.933333333
    },
    "Ester": {
        "Accuracy": (94 + 827) / (94 + 4 + 8 + 827),  # 0.9871
        "Precision": 0.921568627,
        "Recall": 0.959183673,
        "F1": 0.94
    },
    "Ether": {
        "Accuracy": (211 + 687) / (211 + 17 + 18 + 687),  # 0.9624
        "Precision": 0.92139738,
        "Recall": 0.925438596,
        "F1": 0.923413567
    },
    "Ketone": {
        "Accuracy": (85 + 823) / (85 + 13 + 12 + 823),  # 0.9732
        "Precision": 0.87628866,
        "Recall": 0.867346939,
        "F1": 0.871794872
    },
    "Nitrile": {
        "Accuracy": (19 + 883) / (19 + 15 + 16 + 883),  # 0.9668
        "Precision": 0.542857143,
        "Recall": 0.558823529,
        "F1": 0.550724638
    },
    "Nitro": {
        "Accuracy": (36 + 892) / (36 + 0 + 5 + 892),  # 0.9946
        "Precision": 0.87804878,
        "Recall": 1.0,
        "F1": 0.935064935
    }
}

JJ_PERFORMANCE = {
    "Alkane": {
        "Accuracy": 0.992,
        "Precision": 0.994,
        "Recall": 0.998,
        "F1": 0.996,
        "AUC": 0.967
    },
    "Alcohol": {
        "Accuracy": 0.985,
        "Precision": 0.977,
        "Recall": 0.982,
        "F1": 0.980,
        "AUC": 0.985
    },
    "Alkene": {
        "Accuracy": 0.974,
        "Precision": 0.913,
        "Recall": 0.913,
        "F1": 0.913,
        "AUC": 0.949
    },
    "Alkyne": {
        "Accuracy": 0.998,
        "Precision": 0.972,
        "Recall": 0.986,
        "F1": 0.979,
        "AUC": 0.992
    },
    "Amine": {
        "Accuracy": 0.986,
        "Precision": 0.985,
        "Recall": 0.935,
        "F1": 0.959,
        "AUC": 0.966
    },
    "Aromatic": {
        "Accuracy": 0.993,
        "Precision": 0.991,
        "Recall": 0.991,
        "F1": 0.991,
        "AUC": 0.993
    },
    "Carboxylic Acid": {
        "Accuracy": 0.994,
        "Precision": 0.984,
        "Recall": 0.938,
        "F1": 0.960,
        "AUC": 0.968
    },
    "Ester": {
        "Accuracy": 0.989,
        "Precision": 0.905,
        "Recall": 0.971,
        "F1": 0.937,
        "AUC": 0.981
    },
    "Ether": {
        "Accuracy": 0.987,
        "Precision": 0.960,
        "Recall": 0.963,
        "F1": 0.961,
        "AUC": 0.977
    },
    "Aldehyde": {
        "Accuracy": 0.992,
        "Precision": 0.893,
        "Recall": 0.906,
        "F1": 0.899,
        "AUC": 0.951
    },
    "Ketone": {
        "Accuracy": 0.984,
        "Precision": 0.935,
        "Recall": 0.899,
        "F1": 0.916,
        "AUC": 0.946
    },
    "Nitrile": {
        "Accuracy": 0.994,
        "Precision": 0.918,
        "Recall": 0.963,
        "F1": 0.940,
        "AUC": 0.979
    }
}

# Dictionary mapping for easy access
PERFORMANCE_DATA = {
    "aggregate": AGG_PERFORMANCE,
    "mlp": MLP_PERFORMANCE,
    "fcg": FCG_PERFORMANCE,
    "jiangjun": JJ_PERFORMANCE
}