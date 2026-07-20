#!/usr/bin/env python3
"""
Minimal YOLOX-S experiment description for the Model Validation UI.

For .pth checkpoints that need architecture build, install the official YOLOX
package and replace Exp with yolox.exp.yolox_base.Exp (or your custom Exp).

For .onnx weights, only these attributes are required:
  num_classes, test_size, test_conf, nmsthre
"""


class Exp:
    def __init__(self):
        self.num_classes = 80
        self.depth = 0.33
        self.width = 0.50
        self.name = "yolox-s"
        self.test_size = (640, 640)
        self.test_conf = 0.25
        self.nmsthre = 0.45
        self.p6 = False
        self.legacy = False
