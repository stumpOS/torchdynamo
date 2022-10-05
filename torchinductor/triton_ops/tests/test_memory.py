

import torch
from torch import tensor, device
import torch.fx as fx
from torchdynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx

# torch version: 1.13.0a0+git071f875
# torch cuda version: 11.6
# torch git version: 071f875046202b87213865dfc180abdf8368f116


# CUDA Info:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2022 NVIDIA Corporation
# Built on Thu_Feb_10_18:23:41_PST_2022
# Cuda compilation tools, release 11.6, V11.6.112
# Build cuda_11.6.r11.6/compiler.30978841_0

# GPU Hardware Info:
# NVIDIA A100-SXM4-40GB : 8


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()



    def forward(self, primals_3, primals_4, primals_6, primals_7, primals_9, primals_10, primals_12, primals_14, primals_15, primals_21, primals_22, primals_27, primals_28, primals_34, primals_35, primals_40, primals_41, primals_47, primals_48, primals_53, primals_54, primals_60, primals_61, primals_66, primals_68, primals_69, primals_73, primals_74, primals_79, primals_80, primals_84, primals_85, primals_90, primals_91, primals_95, primals_96, primals_101, primals_102, primals_106, primals_107, primals_112, primals_113, primals_117, primals_118, primals_123, primals_124, primals_128, primals_129, primals_134, primals_135, primals_139, primals_140, primals_145, primals_146, primals_150, primals_151, primals_156, primals_157, primals_161, primals_162, primals_167, primals_168, primals_172, primals_173, primals_178, primals_179, primals_183, primals_184, primals_189, primals_190, primals_194, primals_195, primals_200, primals_201, primals_205, primals_206, primals_211, primals_212, primals_216, primals_217, primals_222, primals_223, primals_228, primals_229, primals_234, primals_235, primals_240, primals_241, primals_246, primals_261, convolution, squeeze_5, relu, convolution_1, squeeze_11, relu_1, convolution_2, squeeze_17, relu_2, clone, getitem_1, reciprocal_3, view_2, div, _unsafe_view_6, clone_6, getitem_3, reciprocal_4, view_7, clone_7, getitem_5, reciprocal_6, view_10, div_1, _unsafe_view_16, clone_13, getitem_7, reciprocal_7, view_15, clone_14, getitem_9, reciprocal_9, view_18, div_2, _unsafe_view_26, clone_20, getitem_11, reciprocal_10, view_23, clone_21, getitem_13, reciprocal_12, view_26, div_3, _unsafe_view_36, clone_27, getitem_15, reciprocal_13, view_31, permute_57, clone_28, getitem_17, reciprocal_15, div_4, view_35, clone_33, getitem_22, reciprocal_16, view_37, clone_34, getitem_24, reciprocal_18, div_5, view_41, clone_39, getitem_29, reciprocal_19, view_43, clone_40, getitem_31, reciprocal_21, div_6, view_47, clone_45, getitem_36, reciprocal_22, view_49, clone_46, getitem_38, reciprocal_24, div_7, view_53, clone_51, getitem_43, reciprocal_25, view_55, clone_52, getitem_45, reciprocal_27, div_8, view_59, clone_57, getitem_50, reciprocal_28, view_61, clone_58, getitem_52, reciprocal_30, div_9, view_65, clone_63, getitem_57, reciprocal_31, view_67, clone_64, getitem_59, reciprocal_33, div_10, view_71, clone_69, getitem_64, reciprocal_34, view_73, clone_70, getitem_66, reciprocal_36, div_11, view_77, clone_75, getitem_71, reciprocal_37, view_79, clone_76, getitem_73, reciprocal_39, div_12, view_83, clone_81, getitem_78, reciprocal_40, view_85, clone_82, getitem_80, reciprocal_42, div_13, view_89, clone_87, getitem_85, reciprocal_43, view_91, clone_88, getitem_87, reciprocal_45, div_14, view_95, clone_93, getitem_92, reciprocal_46, view_97, clone_94, getitem_94, reciprocal_48, div_15, view_101, clone_99, getitem_99, reciprocal_49, view_103, clone_100, getitem_101, reciprocal_51, div_16, view_107, clone_105, getitem_106, reciprocal_52, view_109, clone_106, getitem_108, reciprocal_54, div_17, view_113, clone_111, getitem_113, reciprocal_55, view_115, cat, getitem_115, reciprocal_57, view_119, bmm_32, amax_18, sum_19, view_124, add_308, getitem_119, reciprocal_58, view_128, cat_1, getitem_121, reciprocal_60, view_132, bmm_34, amax_19, sum_20, view_137, add_320, getitem_125, reciprocal_61, view_141, cat_2, getitem_127, reciprocal_63, select, _unsafe_view_192, unsqueeze_61, permute_177, permute_179, permute_183, add_341, permute_187, permute_191, permute_197, permute_198, permute_199, permute_203, permute_208, permute_210, add_352, permute_214, permute_218, permute_224, permute_225, permute_226, permute_230, permute_235, permute_239, add_363, permute_243, permute_247, permute_251, permute_252, permute_253, permute_258, permute_262, add_372, permute_266, permute_270, permute_274, permute_275, permute_276, permute_281, permute_285, add_381, permute_289, permute_293, permute_297, permute_298, permute_299, permute_304, permute_308, add_390, permute_312, permute_316, permute_320, permute_321, permute_322, permute_327, permute_331, add_399, permute_335, permute_339, permute_343, permute_344, permute_345, permute_350, permute_354, add_408, permute_358, permute_362, permute_366, permute_367, permute_368, permute_373, permute_377, add_417, permute_381, permute_385, permute_389, permute_390, permute_391, permute_396, permute_400, add_426, permute_404, permute_408, permute_412, permute_413, permute_414, permute_419, permute_423, add_435, permute_427, permute_431, permute_435, permute_436, permute_437, permute_442, permute_446, add_444, permute_450, permute_454, permute_458, permute_459, permute_460, permute_465, permute_469, add_453, permute_473, permute_477, permute_481, permute_482, permute_483, permute_488, permute_492, add_462, permute_496, permute_500, permute_504, permute_505, permute_506, permute_511, permute_515, add_471, permute_519, permute_523, permute_527, permute_528, permute_529, permute_534, permute_538, add_480, permute_542, permute_546, permute_550, permute_551, permute_552, permute_557, permute_563, add_489, permute_567, permute_571, permute_576, permute_580, permute_588, permute_592, add_499, permute_596, permute_600, permute_605, permute_609, permute_617, permute_621, add_509, permute_625, permute_629, permute_634, permute_638, permute_646, permute_650, add_519, permute_654, permute_658, permute_663, permute_667, permute_675, unsqueeze_64, unsqueeze_76, unsqueeze_88, tangents_1):
        sub_3 = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = getitem_1 = None
        mul_21 = torch.ops.aten.mul.Tensor(sub_3, reciprocal_3);  sub_3 = None
        mul_22 = torch.ops.aten.mul.Tensor(mul_21, primals_14)
        add_16 = torch.ops.aten.add.Tensor(mul_22, primals_15);  mul_22 = primals_15 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(add_16, torch.float32);  add_16 = None
        view = torch.ops.aten.view.default(convert_element_type, [50176, 192])
        permute_5 = torch.ops.aten.permute.default(convert_element_type, [0, 3, 1, 2]);  convert_element_type = None
        alias_14 = torch.ops.aten.alias.default(div)
        alias_15 = torch.ops.aten.alias.default(alias_14);  alias_14 = None
        expand = torch.ops.aten.expand.default(div, [64, 6, 196, 9, 9]);  div = None
        view_4 = torch.ops.aten.view.default(expand, [75264, 9, 9]);  expand = None
        sub_5 = torch.ops.aten.sub.Tensor(clone_6, getitem_3);  clone_6 = getitem_3 = None
        mul_24 = torch.ops.aten.mul.Tensor(sub_5, reciprocal_4);  sub_5 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, primals_21)
        add_25 = torch.ops.aten.add.Tensor(mul_25, primals_22);  mul_25 = primals_22 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(add_25, torch.float32);  add_25 = None
        view_6 = torch.ops.aten.view.default(convert_element_type_1, [50176, 192]);  convert_element_type_1 = None
        sub_7 = torch.ops.aten.sub.Tensor(clone_7, getitem_5);  clone_7 = getitem_5 = None
        mul_39 = torch.ops.aten.mul.Tensor(sub_7, reciprocal_6);  sub_7 = None
        mul_40 = torch.ops.aten.mul.Tensor(mul_39, primals_27)
        add_36 = torch.ops.aten.add.Tensor(mul_40, primals_28);  mul_40 = primals_28 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(add_36, torch.float32);  add_36 = None
        view_8 = torch.ops.aten.view.default(convert_element_type_2, [50176, 192])
        permute_19 = torch.ops.aten.permute.default(convert_element_type_2, [0, 3, 1, 2]);  convert_element_type_2 = None
        alias_25 = torch.ops.aten.alias.default(div_1)
        alias_26 = torch.ops.aten.alias.default(alias_25);  alias_25 = None
        expand_2 = torch.ops.aten.expand.default(div_1, [64, 6, 196, 9, 9]);  div_1 = None
        view_12 = torch.ops.aten.view.default(expand_2, [75264, 9, 9]);  expand_2 = None
        sub_9 = torch.ops.aten.sub.Tensor(clone_13, getitem_7);  clone_13 = getitem_7 = None
        mul_42 = torch.ops.aten.mul.Tensor(sub_9, reciprocal_7);  sub_9 = None
        mul_43 = torch.ops.aten.mul.Tensor(mul_42, primals_34)
        add_45 = torch.ops.aten.add.Tensor(mul_43, primals_35);  mul_43 = primals_35 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(add_45, torch.float32);  add_45 = None
        view_14 = torch.ops.aten.view.default(convert_element_type_3, [50176, 192]);  convert_element_type_3 = None
        sub_11 = torch.ops.aten.sub.Tensor(clone_14, getitem_9);  clone_14 = getitem_9 = None
        mul_57 = torch.ops.aten.mul.Tensor(sub_11, reciprocal_9);  sub_11 = None
        mul_58 = torch.ops.aten.mul.Tensor(mul_57, primals_40)
        add_56 = torch.ops.aten.add.Tensor(mul_58, primals_41);  mul_58 = primals_41 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(add_56, torch.float32);  add_56 = None
        view_16 = torch.ops.aten.view.default(convert_element_type_4, [50176, 192])
        permute_33 = torch.ops.aten.permute.default(convert_element_type_4, [0, 3, 1, 2]);  convert_element_type_4 = None
        alias_36 = torch.ops.aten.alias.default(div_2)
        alias_37 = torch.ops.aten.alias.default(alias_36);  alias_36 = None
        expand_4 = torch.ops.aten.expand.default(div_2, [64, 6, 196, 9, 9]);  div_2 = None
        view_20 = torch.ops.aten.view.default(expand_4, [75264, 9, 9]);  expand_4 = None
        sub_13 = torch.ops.aten.sub.Tensor(clone_20, getitem_11);  clone_20 = getitem_11 = None
        mul_60 = torch.ops.aten.mul.Tensor(sub_13, reciprocal_10);  sub_13 = None
        mul_61 = torch.ops.aten.mul.Tensor(mul_60, primals_47)
        add_65 = torch.ops.aten.add.Tensor(mul_61, primals_48);  mul_61 = primals_48 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(add_65, torch.float32);  add_65 = None
        view_22 = torch.ops.aten.view.default(convert_element_type_5, [50176, 192]);  convert_element_type_5 = None
        sub_15 = torch.ops.aten.sub.Tensor(clone_21, getitem_13);  clone_21 = getitem_13 = None
        mul_75 = torch.ops.aten.mul.Tensor(sub_15, reciprocal_12);  sub_15 = None
        mul_76 = torch.ops.aten.mul.Tensor(mul_75, primals_53)
        add_76 = torch.ops.aten.add.Tensor(mul_76, primals_54);  mul_76 = primals_54 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(add_76, torch.float32);  add_76 = None
        view_24 = torch.ops.aten.view.default(convert_element_type_6, [50176, 192])
        permute_47 = torch.ops.aten.permute.default(convert_element_type_6, [0, 3, 1, 2]);  convert_element_type_6 = None
        alias_47 = torch.ops.aten.alias.default(div_3)
        alias_48 = torch.ops.aten.alias.default(alias_47);  alias_47 = None
        expand_6 = torch.ops.aten.expand.default(div_3, [64, 6, 196, 9, 9]);  div_3 = None
        view_28 = torch.ops.aten.view.default(expand_6, [75264, 9, 9]);  expand_6 = None
        sub_17 = torch.ops.aten.sub.Tensor(clone_27, getitem_15);  clone_27 = getitem_15 = None
        mul_78 = torch.ops.aten.mul.Tensor(sub_17, reciprocal_13);  sub_17 = None
        mul_79 = torch.ops.aten.mul.Tensor(mul_78, primals_60)
        add_85 = torch.ops.aten.add.Tensor(mul_79, primals_61);  mul_79 = primals_61 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(add_85, torch.float32);  add_85 = None
        view_30 = torch.ops.aten.view.default(convert_element_type_7, [50176, 192]);  convert_element_type_7 = None
        sub_19 = torch.ops.aten.sub.Tensor(clone_28, getitem_17);  clone_28 = getitem_17 = None
        mul_93 = torch.ops.aten.mul.Tensor(sub_19, reciprocal_15);  sub_19 = None
        mul_94 = torch.ops.aten.mul.Tensor(mul_93, primals_68)
        add_97 = torch.ops.aten.add.Tensor(mul_94, primals_69);  mul_94 = primals_69 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(add_97, torch.float32);  add_97 = None
        view_32 = torch.ops.aten.view.default(convert_element_type_8, [12544, 384]);  convert_element_type_8 = None
        alias_54 = torch.ops.aten.alias.default(div_4)
        alias_55 = torch.ops.aten.alias.default(alias_54);  alias_54 = None
        expand_10 = torch.ops.aten.expand.default(div_4, [64, 12, 196, 196]);  div_4 = None
        view_34 = torch.ops.aten.view.default(expand_10, [768, 196, 196]);  expand_10 = None
        sub_21 = torch.ops.aten.sub.Tensor(clone_33, getitem_22);  clone_33 = getitem_22 = None
        mul_96 = torch.ops.aten.mul.Tensor(sub_21, reciprocal_16);  sub_21 = None
        mul_97 = torch.ops.aten.mul.Tensor(mul_96, primals_73)
        add_101 = torch.ops.aten.add.Tensor(mul_97, primals_74);  mul_97 = primals_74 = None
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(add_101, torch.float32);  add_101 = None
        view_36 = torch.ops.aten.view.default(convert_element_type_9, [12544, 384]);  convert_element_type_9 = None
        sub_23 = torch.ops.aten.sub.Tensor(clone_34, getitem_24);  clone_34 = getitem_24 = None
        mul_111 = torch.ops.aten.mul.Tensor(sub_23, reciprocal_18);  sub_23 = None
        mul_112 = torch.ops.aten.mul.Tensor(mul_111, primals_79)
        add_112 = torch.ops.aten.add.Tensor(mul_112, primals_80);  mul_112 = primals_80 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(add_112, torch.float32);  add_112 = None
        view_38 = torch.ops.aten.view.default(convert_element_type_10, [12544, 384]);  convert_element_type_10 = None
        alias_57 = torch.ops.aten.alias.default(div_5)
        alias_58 = torch.ops.aten.alias.default(alias_57);  alias_57 = None
        expand_14 = torch.ops.aten.expand.default(div_5, [64, 12, 196, 196]);  div_5 = None
        view_40 = torch.ops.aten.view.default(expand_14, [768, 196, 196]);  expand_14 = None
        sub_25 = torch.ops.aten.sub.Tensor(clone_39, getitem_29);  clone_39 = getitem_29 = None
        mul_114 = torch.ops.aten.mul.Tensor(sub_25, reciprocal_19);  sub_25 = None
        mul_115 = torch.ops.aten.mul.Tensor(mul_114, primals_84)
        add_116 = torch.ops.aten.add.Tensor(mul_115, primals_85);  mul_115 = primals_85 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(add_116, torch.float32);  add_116 = None
        view_42 = torch.ops.aten.view.default(convert_element_type_11, [12544, 384]);  convert_element_type_11 = None
        sub_27 = torch.ops.aten.sub.Tensor(clone_40, getitem_31);  clone_40 = getitem_31 = None
        mul_129 = torch.ops.aten.mul.Tensor(sub_27, reciprocal_21);  sub_27 = None
        mul_130 = torch.ops.aten.mul.Tensor(mul_129, primals_90)
        add_127 = torch.ops.aten.add.Tensor(mul_130, primals_91);  mul_130 = primals_91 = None
        convert_element_type_12 = torch.ops.prims.convert_element_type.default(add_127, torch.float32);  add_127 = None
        view_44 = torch.ops.aten.view.default(convert_element_type_12, [12544, 384]);  convert_element_type_12 = None
        alias_60 = torch.ops.aten.alias.default(div_6)
        alias_61 = torch.ops.aten.alias.default(alias_60);  alias_60 = None
        expand_18 = torch.ops.aten.expand.default(div_6, [64, 12, 196, 196]);  div_6 = None
        view_46 = torch.ops.aten.view.default(expand_18, [768, 196, 196]);  expand_18 = None
        sub_29 = torch.ops.aten.sub.Tensor(clone_45, getitem_36);  clone_45 = getitem_36 = None
        mul_132 = torch.ops.aten.mul.Tensor(sub_29, reciprocal_22);  sub_29 = None
        mul_133 = torch.ops.aten.mul.Tensor(mul_132, primals_95)
        add_131 = torch.ops.aten.add.Tensor(mul_133, primals_96);  mul_133 = primals_96 = None
        convert_element_type_13 = torch.ops.prims.convert_element_type.default(add_131, torch.float32);  add_131 = None
        view_48 = torch.ops.aten.view.default(convert_element_type_13, [12544, 384]);  convert_element_type_13 = None
        sub_31 = torch.ops.aten.sub.Tensor(clone_46, getitem_38);  clone_46 = getitem_38 = None
        mul_147 = torch.ops.aten.mul.Tensor(sub_31, reciprocal_24);  sub_31 = None
        mul_148 = torch.ops.aten.mul.Tensor(mul_147, primals_101)
        add_142 = torch.ops.aten.add.Tensor(mul_148, primals_102);  mul_148 = primals_102 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(add_142, torch.float32);  add_142 = None
        view_50 = torch.ops.aten.view.default(convert_element_type_14, [12544, 384]);  convert_element_type_14 = None
        alias_63 = torch.ops.aten.alias.default(div_7)
        alias_64 = torch.ops.aten.alias.default(alias_63);  alias_63 = None
        expand_22 = torch.ops.aten.expand.default(div_7, [64, 12, 196, 196]);  div_7 = None
        view_52 = torch.ops.aten.view.default(expand_22, [768, 196, 196]);  expand_22 = None
        sub_33 = torch.ops.aten.sub.Tensor(clone_51, getitem_43);  clone_51 = getitem_43 = None
        mul_150 = torch.ops.aten.mul.Tensor(sub_33, reciprocal_25);  sub_33 = None
        mul_151 = torch.ops.aten.mul.Tensor(mul_150, primals_106)
        add_146 = torch.ops.aten.add.Tensor(mul_151, primals_107);  mul_151 = primals_107 = None
        convert_element_type_15 = torch.ops.prims.convert_element_type.default(add_146, torch.float32);  add_146 = None
        view_54 = torch.ops.aten.view.default(convert_element_type_15, [12544, 384]);  convert_element_type_15 = None
        sub_35 = torch.ops.aten.sub.Tensor(clone_52, getitem_45);  clone_52 = getitem_45 = None
        mul_165 = torch.ops.aten.mul.Tensor(sub_35, reciprocal_27);  sub_35 = None
        mul_166 = torch.ops.aten.mul.Tensor(mul_165, primals_112)
        add_157 = torch.ops.aten.add.Tensor(mul_166, primals_113);  mul_166 = primals_113 = None
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(add_157, torch.float32);  add_157 = None
        view_56 = torch.ops.aten.view.default(convert_element_type_16, [12544, 384]);  convert_element_type_16 = None
        alias_66 = torch.ops.aten.alias.default(div_8)
        alias_67 = torch.ops.aten.alias.default(alias_66);  alias_66 = None
        expand_26 = torch.ops.aten.expand.default(div_8, [64, 12, 196, 196]);  div_8 = None
        view_58 = torch.ops.aten.view.default(expand_26, [768, 196, 196]);  expand_26 = None
        sub_37 = torch.ops.aten.sub.Tensor(clone_57, getitem_50);  clone_57 = getitem_50 = None
        mul_168 = torch.ops.aten.mul.Tensor(sub_37, reciprocal_28);  sub_37 = None
        mul_169 = torch.ops.aten.mul.Tensor(mul_168, primals_117)
        add_161 = torch.ops.aten.add.Tensor(mul_169, primals_118);  mul_169 = primals_118 = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(add_161, torch.float32);  add_161 = None
        view_60 = torch.ops.aten.view.default(convert_element_type_17, [12544, 384]);  convert_element_type_17 = None
        sub_39 = torch.ops.aten.sub.Tensor(clone_58, getitem_52);  clone_58 = getitem_52 = None
        mul_183 = torch.ops.aten.mul.Tensor(sub_39, reciprocal_30);  sub_39 = None
        mul_184 = torch.ops.aten.mul.Tensor(mul_183, primals_123)
        add_172 = torch.ops.aten.add.Tensor(mul_184, primals_124);  mul_184 = primals_124 = None
        convert_element_type_18 = torch.ops.prims.convert_element_type.default(add_172, torch.float32);  add_172 = None
        view_62 = torch.ops.aten.view.default(convert_element_type_18, [12544, 384]);  convert_element_type_18 = None
        alias_69 = torch.ops.aten.alias.default(div_9)
        alias_70 = torch.ops.aten.alias.default(alias_69);  alias_69 = None
        expand_30 = torch.ops.aten.expand.default(div_9, [64, 12, 196, 196]);  div_9 = None
        view_64 = torch.ops.aten.view.default(expand_30, [768, 196, 196]);  expand_30 = None
        sub_41 = torch.ops.aten.sub.Tensor(clone_63, getitem_57);  clone_63 = getitem_57 = None
        mul_186 = torch.ops.aten.mul.Tensor(sub_41, reciprocal_31);  sub_41 = None
        mul_187 = torch.ops.aten.mul.Tensor(mul_186, primals_128)
        add_176 = torch.ops.aten.add.Tensor(mul_187, primals_129);  mul_187 = primals_129 = None
        convert_element_type_19 = torch.ops.prims.convert_element_type.default(add_176, torch.float32);  add_176 = None
        view_66 = torch.ops.aten.view.default(convert_element_type_19, [12544, 384]);  convert_element_type_19 = None
        sub_43 = torch.ops.aten.sub.Tensor(clone_64, getitem_59);  clone_64 = getitem_59 = None
        mul_201 = torch.ops.aten.mul.Tensor(sub_43, reciprocal_33);  sub_43 = None
        mul_202 = torch.ops.aten.mul.Tensor(mul_201, primals_134)
        add_187 = torch.ops.aten.add.Tensor(mul_202, primals_135);  mul_202 = primals_135 = None
        convert_element_type_20 = torch.ops.prims.convert_element_type.default(add_187, torch.float32);  add_187 = None
        view_68 = torch.ops.aten.view.default(convert_element_type_20, [12544, 384]);  convert_element_type_20 = None
        alias_72 = torch.ops.aten.alias.default(div_10)
        alias_73 = torch.ops.aten.alias.default(alias_72);  alias_72 = None
        expand_34 = torch.ops.aten.expand.default(div_10, [64, 12, 196, 196]);  div_10 = None
        view_70 = torch.ops.aten.view.default(expand_34, [768, 196, 196]);  expand_34 = None
        sub_45 = torch.ops.aten.sub.Tensor(clone_69, getitem_64);  clone_69 = getitem_64 = None
        mul_204 = torch.ops.aten.mul.Tensor(sub_45, reciprocal_34);  sub_45 = None
        mul_205 = torch.ops.aten.mul.Tensor(mul_204, primals_139)
        add_191 = torch.ops.aten.add.Tensor(mul_205, primals_140);  mul_205 = primals_140 = None
        convert_element_type_21 = torch.ops.prims.convert_element_type.default(add_191, torch.float32);  add_191 = None
        view_72 = torch.ops.aten.view.default(convert_element_type_21, [12544, 384]);  convert_element_type_21 = None
        sub_47 = torch.ops.aten.sub.Tensor(clone_70, getitem_66);  clone_70 = getitem_66 = None
        mul_219 = torch.ops.aten.mul.Tensor(sub_47, reciprocal_36);  sub_47 = None
        mul_220 = torch.ops.aten.mul.Tensor(mul_219, primals_145)
        add_202 = torch.ops.aten.add.Tensor(mul_220, primals_146);  mul_220 = primals_146 = None
        convert_element_type_22 = torch.ops.prims.convert_element_type.default(add_202, torch.float32);  add_202 = None
        view_74 = torch.ops.aten.view.default(convert_element_type_22, [12544, 384]);  convert_element_type_22 = None
        alias_75 = torch.ops.aten.alias.default(div_11)
        alias_76 = torch.ops.aten.alias.default(alias_75);  alias_75 = None
        expand_38 = torch.ops.aten.expand.default(div_11, [64, 12, 196, 196]);  div_11 = None
        view_76 = torch.ops.aten.view.default(expand_38, [768, 196, 196]);  expand_38 = None
        sub_49 = torch.ops.aten.sub.Tensor(clone_75, getitem_71);  clone_75 = getitem_71 = None
        mul_222 = torch.ops.aten.mul.Tensor(sub_49, reciprocal_37);  sub_49 = None
        mul_223 = torch.ops.aten.mul.Tensor(mul_222, primals_150)
        add_206 = torch.ops.aten.add.Tensor(mul_223, primals_151);  mul_223 = primals_151 = None
        convert_element_type_23 = torch.ops.prims.convert_element_type.default(add_206, torch.float32);  add_206 = None
        view_78 = torch.ops.aten.view.default(convert_element_type_23, [12544, 384]);  convert_element_type_23 = None
        sub_51 = torch.ops.aten.sub.Tensor(clone_76, getitem_73);  clone_76 = getitem_73 = None
        mul_237 = torch.ops.aten.mul.Tensor(sub_51, reciprocal_39);  sub_51 = None
        mul_238 = torch.ops.aten.mul.Tensor(mul_237, primals_156)
        add_217 = torch.ops.aten.add.Tensor(mul_238, primals_157);  mul_238 = primals_157 = None
        convert_element_type_24 = torch.ops.prims.convert_element_type.default(add_217, torch.float32);  add_217 = None
        view_80 = torch.ops.aten.view.default(convert_element_type_24, [12544, 384]);  convert_element_type_24 = None
        alias_78 = torch.ops.aten.alias.default(div_12)
        alias_79 = torch.ops.aten.alias.default(alias_78);  alias_78 = None
        expand_42 = torch.ops.aten.expand.default(div_12, [64, 12, 196, 196]);  div_12 = None
        view_82 = torch.ops.aten.view.default(expand_42, [768, 196, 196]);  expand_42 = None
        sub_53 = torch.ops.aten.sub.Tensor(clone_81, getitem_78);  clone_81 = getitem_78 = None
        mul_240 = torch.ops.aten.mul.Tensor(sub_53, reciprocal_40);  sub_53 = None
        mul_241 = torch.ops.aten.mul.Tensor(mul_240, primals_161)
        add_221 = torch.ops.aten.add.Tensor(mul_241, primals_162);  mul_241 = primals_162 = None
        convert_element_type_25 = torch.ops.prims.convert_element_type.default(add_221, torch.float32);  add_221 = None
        view_84 = torch.ops.aten.view.default(convert_element_type_25, [12544, 384]);  convert_element_type_25 = None
        sub_55 = torch.ops.aten.sub.Tensor(clone_82, getitem_80);  clone_82 = getitem_80 = None
        mul_255 = torch.ops.aten.mul.Tensor(sub_55, reciprocal_42);  sub_55 = None
        mul_256 = torch.ops.aten.mul.Tensor(mul_255, primals_167)
        add_232 = torch.ops.aten.add.Tensor(mul_256, primals_168);  mul_256 = primals_168 = None
        convert_element_type_26 = torch.ops.prims.convert_element_type.default(add_232, torch.float32);  add_232 = None
        view_86 = torch.ops.aten.view.default(convert_element_type_26, [12544, 384]);  convert_element_type_26 = None
        alias_81 = torch.ops.aten.alias.default(div_13)
        alias_82 = torch.ops.aten.alias.default(alias_81);  alias_81 = None
        expand_46 = torch.ops.aten.expand.default(div_13, [64, 12, 196, 196]);  div_13 = None
        view_88 = torch.ops.aten.view.default(expand_46, [768, 196, 196]);  expand_46 = None
        sub_57 = torch.ops.aten.sub.Tensor(clone_87, getitem_85);  clone_87 = getitem_85 = None
        mul_258 = torch.ops.aten.mul.Tensor(sub_57, reciprocal_43);  sub_57 = None
        mul_259 = torch.ops.aten.mul.Tensor(mul_258, primals_172)
        add_236 = torch.ops.aten.add.Tensor(mul_259, primals_173);  mul_259 = primals_173 = None
        convert_element_type_27 = torch.ops.prims.convert_element_type.default(add_236, torch.float32);  add_236 = None
        view_90 = torch.ops.aten.view.default(convert_element_type_27, [12544, 384]);  convert_element_type_27 = None
        sub_59 = torch.ops.aten.sub.Tensor(clone_88, getitem_87);  clone_88 = getitem_87 = None
        mul_273 = torch.ops.aten.mul.Tensor(sub_59, reciprocal_45);  sub_59 = None
        mul_274 = torch.ops.aten.mul.Tensor(mul_273, primals_178)
        add_247 = torch.ops.aten.add.Tensor(mul_274, primals_179);  mul_274 = primals_179 = None
        convert_element_type_28 = torch.ops.prims.convert_element_type.default(add_247, torch.float32);  add_247 = None
        view_92 = torch.ops.aten.view.default(convert_element_type_28, [12544, 384]);  convert_element_type_28 = None
        alias_84 = torch.ops.aten.alias.default(div_14)
        alias_85 = torch.ops.aten.alias.default(alias_84);  alias_84 = None
        expand_50 = torch.ops.aten.expand.default(div_14, [64, 12, 196, 196]);  div_14 = None
        view_94 = torch.ops.aten.view.default(expand_50, [768, 196, 196]);  expand_50 = None
        sub_61 = torch.ops.aten.sub.Tensor(clone_93, getitem_92);  clone_93 = getitem_92 = None
        mul_276 = torch.ops.aten.mul.Tensor(sub_61, reciprocal_46);  sub_61 = None
        mul_277 = torch.ops.aten.mul.Tensor(mul_276, primals_183)
        add_251 = torch.ops.aten.add.Tensor(mul_277, primals_184);  mul_277 = primals_184 = None
        convert_element_type_29 = torch.ops.prims.convert_element_type.default(add_251, torch.float32);  add_251 = None
        view_96 = torch.ops.aten.view.default(convert_element_type_29, [12544, 384]);  convert_element_type_29 = None
        sub_63 = torch.ops.aten.sub.Tensor(clone_94, getitem_94);  clone_94 = getitem_94 = None
        mul_291 = torch.ops.aten.mul.Tensor(sub_63, reciprocal_48);  sub_63 = None
        mul_292 = torch.ops.aten.mul.Tensor(mul_291, primals_189)
        add_262 = torch.ops.aten.add.Tensor(mul_292, primals_190);  mul_292 = primals_190 = None
        convert_element_type_30 = torch.ops.prims.convert_element_type.default(add_262, torch.float32);  add_262 = None
        view_98 = torch.ops.aten.view.default(convert_element_type_30, [12544, 384]);  convert_element_type_30 = None
        alias_87 = torch.ops.aten.alias.default(div_15)
        alias_88 = torch.ops.aten.alias.default(alias_87);  alias_87 = None
        expand_54 = torch.ops.aten.expand.default(div_15, [64, 12, 196, 196]);  div_15 = None
        view_100 = torch.ops.aten.view.default(expand_54, [768, 196, 196]);  expand_54 = None
        sub_65 = torch.ops.aten.sub.Tensor(clone_99, getitem_99);  clone_99 = getitem_99 = None
        mul_294 = torch.ops.aten.mul.Tensor(sub_65, reciprocal_49);  sub_65 = None
        mul_295 = torch.ops.aten.mul.Tensor(mul_294, primals_194)
        add_266 = torch.ops.aten.add.Tensor(mul_295, primals_195);  mul_295 = primals_195 = None
        convert_element_type_31 = torch.ops.prims.convert_element_type.default(add_266, torch.float32);  add_266 = None
        view_102 = torch.ops.aten.view.default(convert_element_type_31, [12544, 384]);  convert_element_type_31 = None
        sub_67 = torch.ops.aten.sub.Tensor(clone_100, getitem_101);  clone_100 = getitem_101 = None
        mul_309 = torch.ops.aten.mul.Tensor(sub_67, reciprocal_51);  sub_67 = None
        mul_310 = torch.ops.aten.mul.Tensor(mul_309, primals_200)
        add_277 = torch.ops.aten.add.Tensor(mul_310, primals_201);  mul_310 = primals_201 = None
        convert_element_type_32 = torch.ops.prims.convert_element_type.default(add_277, torch.float32);  add_277 = None
        view_104 = torch.ops.aten.view.default(convert_element_type_32, [12544, 384]);  convert_element_type_32 = None
        alias_90 = torch.ops.aten.alias.default(div_16)
        alias_91 = torch.ops.aten.alias.default(alias_90);  alias_90 = None
        expand_58 = torch.ops.aten.expand.default(div_16, [64, 12, 196, 196]);  div_16 = None
        view_106 = torch.ops.aten.view.default(expand_58, [768, 196, 196]);  expand_58 = None
        sub_69 = torch.ops.aten.sub.Tensor(clone_105, getitem_106);  clone_105 = getitem_106 = None
        mul_312 = torch.ops.aten.mul.Tensor(sub_69, reciprocal_52);  sub_69 = None
        mul_313 = torch.ops.aten.mul.Tensor(mul_312, primals_205)
        add_281 = torch.ops.aten.add.Tensor(mul_313, primals_206);  mul_313 = primals_206 = None
        convert_element_type_33 = torch.ops.prims.convert_element_type.default(add_281, torch.float32);  add_281 = None
        view_108 = torch.ops.aten.view.default(convert_element_type_33, [12544, 384]);  convert_element_type_33 = None
        sub_71 = torch.ops.aten.sub.Tensor(clone_106, getitem_108);  clone_106 = getitem_108 = None
        mul_327 = torch.ops.aten.mul.Tensor(sub_71, reciprocal_54);  sub_71 = None
        mul_328 = torch.ops.aten.mul.Tensor(mul_327, primals_211)
        add_292 = torch.ops.aten.add.Tensor(mul_328, primals_212);  mul_328 = primals_212 = None
        convert_element_type_34 = torch.ops.prims.convert_element_type.default(add_292, torch.float32);  add_292 = None
        view_110 = torch.ops.aten.view.default(convert_element_type_34, [12544, 384]);  convert_element_type_34 = None
        alias_93 = torch.ops.aten.alias.default(div_17)
        alias_94 = torch.ops.aten.alias.default(alias_93);  alias_93 = None
        expand_62 = torch.ops.aten.expand.default(div_17, [64, 12, 196, 196]);  div_17 = None
        view_112 = torch.ops.aten.view.default(expand_62, [768, 196, 196]);  expand_62 = None
        sub_73 = torch.ops.aten.sub.Tensor(clone_111, getitem_113);  clone_111 = getitem_113 = None
        mul_330 = torch.ops.aten.mul.Tensor(sub_73, reciprocal_55);  sub_73 = None
        mul_331 = torch.ops.aten.mul.Tensor(mul_330, primals_216)
        add_296 = torch.ops.aten.add.Tensor(mul_331, primals_217);  mul_331 = primals_217 = None
        convert_element_type_35 = torch.ops.prims.convert_element_type.default(add_296, torch.float32);  add_296 = None
        view_114 = torch.ops.aten.view.default(convert_element_type_35, [12544, 384]);  convert_element_type_35 = None
        sub_75 = torch.ops.aten.sub.Tensor(cat, getitem_115);  cat = getitem_115 = None
        mul_345 = torch.ops.aten.mul.Tensor(sub_75, reciprocal_57);  sub_75 = None
        mul_346 = torch.ops.aten.mul.Tensor(mul_345, primals_222)
        add_307 = torch.ops.aten.add.Tensor(mul_346, primals_223);  mul_346 = primals_223 = None
        convert_element_type_36 = torch.ops.prims.convert_element_type.default(add_307, torch.float32);  add_307 = None
        view_117 = torch.ops.aten.view.default(convert_element_type_36, [12608, 384]);  convert_element_type_36 = None
        _unsafe_view_183 = torch.ops.aten._unsafe_view.default(bmm_32, [64, 12, 1, 197]);  bmm_32 = None
        sub_76 = torch.ops.aten.sub.Tensor(_unsafe_view_183, amax_18);  _unsafe_view_183 = amax_18 = None
        exp_36 = torch.ops.aten.exp.default(sub_76);  sub_76 = None
        div_18 = torch.ops.aten.div.Tensor(exp_36, sum_19);  exp_36 = sum_19 = None
        alias_96 = torch.ops.aten.alias.default(div_18)
        alias_97 = torch.ops.aten.alias.default(alias_96);  alias_96 = None
        expand_67 = torch.ops.aten.expand.default(div_18, [64, 12, 1, 197]);  div_18 = None
        view_122 = torch.ops.aten.view.default(expand_67, [768, 1, 197]);  expand_67 = None
        sub_77 = torch.ops.aten.sub.Tensor(add_308, getitem_119);  add_308 = getitem_119 = None
        mul_348 = torch.ops.aten.mul.Tensor(sub_77, reciprocal_58);  sub_77 = None
        mul_349 = torch.ops.aten.mul.Tensor(mul_348, primals_228)
        add_310 = torch.ops.aten.add.Tensor(mul_349, primals_229);  mul_349 = primals_229 = None
        convert_element_type_37 = torch.ops.prims.convert_element_type.default(add_310, torch.float32);  add_310 = None
        view_126 = torch.ops.aten.view.default(convert_element_type_37, [64, 384]);  convert_element_type_37 = None
        sub_79 = torch.ops.aten.sub.Tensor(cat_1, getitem_121);  cat_1 = getitem_121 = None
        mul_363 = torch.ops.aten.mul.Tensor(sub_79, reciprocal_60);  sub_79 = None
        mul_364 = torch.ops.aten.mul.Tensor(mul_363, primals_234)
        add_319 = torch.ops.aten.add.Tensor(mul_364, primals_235);  mul_364 = primals_235 = None
        convert_element_type_38 = torch.ops.prims.convert_element_type.default(add_319, torch.float32);  add_319 = None
        view_130 = torch.ops.aten.view.default(convert_element_type_38, [12608, 384]);  convert_element_type_38 = None
        _unsafe_view_189 = torch.ops.aten._unsafe_view.default(bmm_34, [64, 12, 1, 197]);  bmm_34 = None
        sub_80 = torch.ops.aten.sub.Tensor(_unsafe_view_189, amax_19);  _unsafe_view_189 = amax_19 = None
        exp_38 = torch.ops.aten.exp.default(sub_80);  sub_80 = None
        div_19 = torch.ops.aten.div.Tensor(exp_38, sum_20);  exp_38 = sum_20 = None
        alias_99 = torch.ops.aten.alias.default(div_19)
        alias_100 = torch.ops.aten.alias.default(alias_99);  alias_99 = None
        expand_71 = torch.ops.aten.expand.default(div_19, [64, 12, 1, 197]);  div_19 = None
        view_135 = torch.ops.aten.view.default(expand_71, [768, 1, 197]);  expand_71 = None
        sub_81 = torch.ops.aten.sub.Tensor(add_320, getitem_125);  add_320 = getitem_125 = None
        mul_366 = torch.ops.aten.mul.Tensor(sub_81, reciprocal_61);  sub_81 = None
        mul_367 = torch.ops.aten.mul.Tensor(mul_366, primals_240)
        add_322 = torch.ops.aten.add.Tensor(mul_367, primals_241);  mul_367 = primals_241 = None
        convert_element_type_39 = torch.ops.prims.convert_element_type.default(add_322, torch.float32);  add_322 = None
        view_139 = torch.ops.aten.view.default(convert_element_type_39, [64, 384]);  convert_element_type_39 = None
        sub_83 = torch.ops.aten.sub.Tensor(cat_2, getitem_127);  cat_2 = getitem_127 = None
        mul_381 = torch.ops.aten.mul.Tensor(sub_83, reciprocal_63);  sub_83 = None
        mul_384 = torch.ops.aten.mul.Tensor(tangents_1, 0.5)
        unsqueeze_60 = torch.ops.aten.unsqueeze.default(mul_384, 1);  mul_384 = None
        zeros = torch.ops.aten.zeros.default([64, 196, 1000], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scatter = torch.ops.aten.scatter.src(zeros, 1, unsqueeze_61, unsqueeze_60);  zeros = unsqueeze_61 = unsqueeze_60 = None
        sum_21 = torch.ops.aten.sum.dim_IntList(scatter, [0, 1], True)
        view_143 = torch.ops.aten.view.default(sum_21, [1000]);  sum_21 = None
        view_144 = torch.ops.aten.view.default(scatter, [12544, 1000]);  scatter = None
        permute_175 = torch.ops.aten.permute.default(view_144, [1, 0])
        mm_81 = torch.ops.aten.mm.default(permute_175, _unsafe_view_192);  permute_175 = _unsafe_view_192 = None
        permute_176 = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
        mm_82 = torch.ops.aten.mm.default(view_144, permute_177);  view_144 = permute_177 = None
        view_145 = torch.ops.aten.view.default(mm_82, [64, 196, 384]);  mm_82 = None
        permute_178 = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
        new_zeros_7 = torch.ops.aten.new_zeros.default(view_145, [64, 197, 384], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter = torch.ops.aten.slice_scatter.default(new_zeros_7, view_145, 1, 1, 9223372036854775807);  new_zeros_7 = view_145 = None
        new_zeros_8 = torch.ops.aten.new_zeros.default(slice_scatter, [64, 197, 384], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_1 = torch.ops.aten.slice_scatter.default(new_zeros_8, slice_scatter, 0, 0, 9223372036854775807);  new_zeros_8 = slice_scatter = None
        mm_83 = torch.ops.aten.mm.default(tangents_1, permute_179);  permute_179 = None
        permute_180 = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_84 = torch.ops.aten.mm.default(permute_180, select);  permute_180 = select = None
        permute_181 = torch.ops.aten.permute.default(mm_84, [1, 0]);  mm_84 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_146 = torch.ops.aten.view.default(sum_22, [1000]);  sum_22 = None
        permute_182 = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
        new_zeros_9 = torch.ops.aten.new_zeros.default(mm_83, [64, 197, 384], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter = torch.ops.aten.select_scatter.default(new_zeros_9, mm_83, 1, 0);  new_zeros_9 = mm_83 = None
        new_zeros_10 = torch.ops.aten.new_zeros.default(select_scatter, [64, 197, 384], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_2 = torch.ops.aten.slice_scatter.default(new_zeros_10, select_scatter, 0, 0, 9223372036854775807);  new_zeros_10 = select_scatter = None
        add_334 = torch.ops.aten.add.Tensor(slice_scatter_1, slice_scatter_2);  slice_scatter_1 = slice_scatter_2 = None
        mul_386 = torch.ops.aten.mul.Tensor(add_334, primals_246);  primals_246 = None
        mul_387 = torch.ops.aten.mul.Tensor(mul_386, 384)
        sum_23 = torch.ops.aten.sum.dim_IntList(mul_386, [2], True)
        mul_388 = torch.ops.aten.mul.Tensor(mul_386, mul_381);  mul_386 = None
        sum_24 = torch.ops.aten.sum.dim_IntList(mul_388, [2], True);  mul_388 = None
        mul_389 = torch.ops.aten.mul.Tensor(mul_381, sum_24);  sum_24 = None
        sub_85 = torch.ops.aten.sub.Tensor(mul_387, sum_23);  mul_387 = sum_23 = None
        sub_86 = torch.ops.aten.sub.Tensor(sub_85, mul_389);  sub_85 = mul_389 = None
        div_20 = torch.ops.aten.div.Tensor(reciprocal_63, 384);  reciprocal_63 = None
        mul_390 = torch.ops.aten.mul.Tensor(div_20, sub_86);  div_20 = sub_86 = None
        mul_391 = torch.ops.aten.mul.Tensor(add_334, mul_381);  mul_381 = None
        sum_25 = torch.ops.aten.sum.dim_IntList(mul_391, [0, 1]);  mul_391 = None
        sum_26 = torch.ops.aten.sum.dim_IntList(add_334, [0, 1]);  add_334 = None
        slice_26 = torch.ops.aten.slice.Tensor(mul_390, 1, 0, 1)
        slice_27 = torch.ops.aten.slice.Tensor(mul_390, 1, 1, 197);  mul_390 = None
        new_zeros_11 = torch.ops.aten.new_zeros.default(slice_27, [64, 197, 384], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_3 = torch.ops.aten.slice_scatter.default(new_zeros_11, slice_27, 1, 1, 9223372036854775807);  new_zeros_11 = slice_27 = None
        new_zeros_12 = torch.ops.aten.new_zeros.default(slice_scatter_3, [64, 197, 384], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_4 = torch.ops.aten.slice_scatter.default(new_zeros_12, slice_scatter_3, 0, 0, 9223372036854775807);  new_zeros_12 = slice_scatter_3 = None
        view_147 = torch.ops.aten.view.default(slice_26, [64, 384])
        mm_85 = torch.ops.aten.mm.default(view_147, permute_183);  permute_183 = None
        permute_184 = torch.ops.aten.permute.default(view_147, [1, 0])
        mm_86 = torch.ops.aten.mm.default(permute_184, view_141);  permute_184 = view_141 = None
        permute_185 = torch.ops.aten.permute.default(mm_86, [1, 0]);  mm_86 = None
        sum_27 = torch.ops.aten.sum.dim_IntList(view_147, [0], True);  view_147 = None
        view_148 = torch.ops.aten.view.default(sum_27, [384]);  sum_27 = None
        view_149 = torch.ops.aten.view.default(mm_85, [64, 1, 1152]);  mm_85 = None
        permute_186 = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
        mul_408 = torch.ops.aten.mul.Tensor(view_149, add_341);  view_149 = add_341 = None
        view_150 = torch.ops.aten.view.default(mul_408, [64, 1152]);  mul_408 = None
        mm_87 = torch.ops.aten.mm.default(view_150, permute_187);  permute_187 = None
        permute_188 = torch.ops.aten.permute.default(view_150, [1, 0])
        mm_88 = torch.ops.aten.mm.default(permute_188, view_139);  permute_188 = view_139 = None
        permute_189 = torch.ops.aten.permute.default(mm_88, [1, 0]);  mm_88 = None
        sum_28 = torch.ops.aten.sum.dim_IntList(view_150, [0], True);  view_150 = None
        view_151 = torch.ops.aten.view.default(sum_28, [1152]);  sum_28 = None
        view_152 = torch.ops.aten.view.default(mm_87, [64, 1, 384]);  mm_87 = None
        permute_190 = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None
        mul_410 = torch.ops.aten.mul.Tensor(view_152, primals_240);  primals_240 = None
        mul_411 = torch.ops.aten.mul.Tensor(mul_410, 384)
        sum_29 = torch.ops.aten.sum.dim_IntList(mul_410, [2], True)
        mul_412 = torch.ops.aten.mul.Tensor(mul_410, mul_366);  mul_410 = None
        sum_30 = torch.ops.aten.sum.dim_IntList(mul_412, [2], True);  mul_412 = None
        mul_413 = torch.ops.aten.mul.Tensor(mul_366, sum_30);  sum_30 = None
        sub_89 = torch.ops.aten.sub.Tensor(mul_411, sum_29);  mul_411 = sum_29 = None
        sub_90 = torch.ops.aten.sub.Tensor(sub_89, mul_413);  sub_89 = mul_413 = None
        div_21 = torch.ops.aten.div.Tensor(reciprocal_61, 384);  reciprocal_61 = None
        mul_414 = torch.ops.aten.mul.Tensor(div_21, sub_90);  div_21 = sub_90 = None
        mul_415 = torch.ops.aten.mul.Tensor(view_152, mul_366);  mul_366 = None
        sum_31 = torch.ops.aten.sum.dim_IntList(mul_415, [0, 1]);  mul_415 = None
        sum_32 = torch.ops.aten.sum.dim_IntList(view_152, [0, 1]);  view_152 = None
        add_342 = torch.ops.aten.add.Tensor(slice_26, mul_414);  slice_26 = mul_414 = None
        view_153 = torch.ops.aten.view.default(add_342, [64, 384])
        mm_89 = torch.ops.aten.mm.default(view_153, permute_191);  permute_191 = None
        permute_192 = torch.ops.aten.permute.default(view_153, [1, 0])
        mm_90 = torch.ops.aten.mm.default(permute_192, view_137);  permute_192 = view_137 = None
        permute_193 = torch.ops.aten.permute.default(mm_90, [1, 0]);  mm_90 = None
        sum_33 = torch.ops.aten.sum.dim_IntList(view_153, [0], True);  view_153 = None
        view_154 = torch.ops.aten.view.default(sum_33, [384]);  sum_33 = None
        view_155 = torch.ops.aten.view.default(mm_89, [64, 1, 384]);  mm_89 = None
        permute_194 = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
        view_156 = torch.ops.aten.view.default(view_155, [64, 1, 12, 32]);  view_155 = None
        permute_195 = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
        view_157 = torch.ops.aten.view.default(permute_195, [768, 1, 32]);  permute_195 = None
        permute_196 = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None
        bmm_36 = torch.ops.aten.bmm.default(permute_196, view_157);  permute_196 = None
        bmm_37 = torch.ops.aten.bmm.default(view_157, permute_197);  view_157 = permute_197 = None
        view_158 = torch.ops.aten.view.default(bmm_36, [64, 12, 197, 32]);  bmm_36 = None
        view_159 = torch.ops.aten.view.default(bmm_37, [64, 12, 1, 197]);  bmm_37 = None
        alias_101 = torch.ops.aten.alias.default(alias_100);  alias_100 = None
        alias_102 = torch.ops.aten.alias.default(alias_101);  alias_101 = None
        mul_416 = torch.ops.aten.mul.Tensor(view_159, alias_102);  view_159 = None
        sum_34 = torch.ops.aten.sum.dim_IntList(mul_416, [-1], True)
        mul_417 = torch.ops.aten.mul.Tensor(alias_102, sum_34);  alias_102 = sum_34 = None
        sub_91 = torch.ops.aten.sub.Tensor(mul_416, mul_417);  mul_416 = mul_417 = None
        view_160 = torch.ops.aten.view.default(sub_91, [768, 1, 197]);  sub_91 = None
        bmm_38 = torch.ops.aten.bmm.default(permute_198, view_160);  permute_198 = None
        bmm_39 = torch.ops.aten.bmm.default(view_160, permute_199);  view_160 = permute_199 = None
        view_161 = torch.ops.aten.view.default(bmm_38, [64, 12, 32, 197]);  bmm_38 = None
        view_162 = torch.ops.aten.view.default(bmm_39, [64, 12, 1, 32]);  bmm_39 = None
        permute_200 = torch.ops.aten.permute.default(view_161, [0, 1, 3, 2]);  view_161 = None
        mul_418 = torch.ops.aten.mul.Tensor(view_162, 0.1767766952966369);  view_162 = None
        view_163 = torch.ops.aten.view.default(mul_418, [64, 1, 384]);  mul_418 = None
        view_164 = torch.ops.aten.view.default(view_163, [64, 384]);  view_163 = None
        permute_201 = torch.ops.aten.permute.default(view_164, [1, 0])
        mm_91 = torch.ops.aten.mm.default(permute_201, view_132);  permute_201 = view_132 = None
        permute_202 = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
        mm_92 = torch.ops.aten.mm.default(view_164, permute_203);  view_164 = permute_203 = None
        view_165 = torch.ops.aten.view.default(mm_92, [64, 1, 384]);  mm_92 = None
        permute_204 = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
        new_zeros_13 = torch.ops.aten.new_zeros.default(view_165, [64, 1, 384], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_5 = torch.ops.aten.slice_scatter.default(new_zeros_13, view_165, 2, 0, 9223372036854775807);  new_zeros_13 = view_165 = None
        new_zeros_14 = torch.ops.aten.new_zeros.default(slice_scatter_5, [64, 197, 384], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_6 = torch.ops.aten.slice_scatter.default(new_zeros_14, slice_scatter_5, 1, 0, 1);  new_zeros_14 = slice_scatter_5 = None
        new_zeros_15 = torch.ops.aten.new_zeros.default(slice_scatter_6, [64, 197, 384], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_7 = torch.ops.aten.slice_scatter.default(new_zeros_15, slice_scatter_6, 0, 0, 9223372036854775807);  new_zeros_15 = slice_scatter_6 = None
        cat_3 = torch.ops.aten.cat.default([permute_200, view_158]);  permute_200 = view_158 = None
        view_166 = torch.ops.aten.view.default(cat_3, [2, 64, 12, 197, 32]);  cat_3 = None
        permute_205 = torch.ops.aten.permute.default(view_166, [1, 3, 0, 2, 4]);  view_166 = None
        clone_117 = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        _unsafe_view_194 = torch.ops.aten._unsafe_view.default(clone_117, [64, 197, 768]);  clone_117 = None
        view_167 = torch.ops.aten.view.default(_unsafe_view_194, [12608, 768]);  _unsafe_view_194 = None
        permute_206 = torch.ops.aten.permute.default(view_167, [1, 0])
        mm_93 = torch.ops.aten.mm.default(permute_206, view_130);  permute_206 = view_130 = None
        permute_207 = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
        mm_94 = torch.ops.aten.mm.default(view_167, permute_208);  view_167 = permute_208 = None
        view_168 = torch.ops.aten.view.default(mm_94, [64, 197, 384]);  mm_94 = None
        add_343 = torch.ops.aten.add.Tensor(slice_scatter_7, view_168);  slice_scatter_7 = view_168 = None
        permute_209 = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
        mul_420 = torch.ops.aten.mul.Tensor(add_343, primals_234);  primals_234 = None
        mul_421 = torch.ops.aten.mul.Tensor(mul_420, 384)
        sum_35 = torch.ops.aten.sum.dim_IntList(mul_420, [2], True)
        mul_422 = torch.ops.aten.mul.Tensor(mul_420, mul_363);  mul_420 = None
        sum_36 = torch.ops.aten.sum.dim_IntList(mul_422, [2], True);  mul_422 = None
        mul_423 = torch.ops.aten.mul.Tensor(mul_363, sum_36);  sum_36 = None
        sub_93 = torch.ops.aten.sub.Tensor(mul_421, sum_35);  mul_421 = sum_35 = None
        sub_94 = torch.ops.aten.sub.Tensor(sub_93, mul_423);  sub_93 = mul_423 = None
        div_22 = torch.ops.aten.div.Tensor(reciprocal_60, 384);  reciprocal_60 = None
        mul_424 = torch.ops.aten.mul.Tensor(div_22, sub_94);  div_22 = sub_94 = None
        mul_425 = torch.ops.aten.mul.Tensor(add_343, mul_363);  mul_363 = None
        sum_37 = torch.ops.aten.sum.dim_IntList(mul_425, [0, 1]);  mul_425 = None
        sum_38 = torch.ops.aten.sum.dim_IntList(add_343, [0, 1]);  add_343 = None
        add_344 = torch.ops.aten.add.Tensor(slice_scatter_4, mul_424);  slice_scatter_4 = mul_424 = None
        new_zeros_16 = torch.ops.aten.new_zeros.default(add_342, [64, 197, 384], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_8 = torch.ops.aten.slice_scatter.default(new_zeros_16, add_342, 1, 0, 1);  new_zeros_16 = add_342 = None
        new_zeros_17 = torch.ops.aten.new_zeros.default(slice_scatter_8, [64, 197, 384], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_9 = torch.ops.aten.slice_scatter.default(new_zeros_17, slice_scatter_8, 0, 0, 9223372036854775807);  new_zeros_17 = slice_scatter_8 = None
        add_345 = torch.ops.aten.add.Tensor(add_344, slice_scatter_9);  add_344 = slice_scatter_9 = None
        slice_28 = torch.ops.aten.slice.Tensor(add_345, 1, 0, 1)
        slice_29 = torch.ops.aten.slice.Tensor(add_345, 1, 1, 197);  add_345 = None
        new_zeros_18 = torch.ops.aten.new_zeros.default(slice_29, [64, 197, 384], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_10 = torch.ops.aten.slice_scatter.default(new_zeros_18, slice_29, 1, 1, 9223372036854775807);  new_zeros_18 = slice_29 = None
        new_zeros_19 = torch.ops.aten.new_zeros.default(slice_scatter_10, [64, 197, 384], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_11 = torch.ops.aten.slice_scatter.default(new_zeros_19, slice_scatter_10, 0, 0, 9223372036854775807);  new_zeros_19 = slice_scatter_10 = None
        view_169 = torch.ops.aten.view.default(slice_28, [64, 384])
        mm_95 = torch.ops.aten.mm.default(view_169, permute_210);  permute_210 = None
        permute_211 = torch.ops.aten.permute.default(view_169, [1, 0])
        mm_96 = torch.ops.aten.mm.default(permute_211, view_128);  permute_211 = view_128 = None
        permute_212 = torch.ops.aten.permute.default(mm_96, [1, 0]);  mm_96 = None
        sum_39 = torch.ops.aten.sum.dim_IntList(view_169, [0], True);  view_169 = None
        view_170 = torch.ops.aten.view.default(sum_39, [384]);  sum_39 = None
        view_171 = torch.ops.aten.view.default(mm_95, [64, 1, 1152]);  mm_95 = None
        permute_213 = torch.ops.aten.permute.default(permute_212, [1, 0]);  permute_212 = None
        mul_442 = torch.ops.aten.mul.Tensor(view_171, add_352);  view_171 = add_352 = None
        view_172 = torch.ops.aten.view.default(mul_442, [64, 1152]);  mul_442 = None
        mm_97 = torch.ops.aten.mm.default(view_172, permute_214);  permute_214 = None
        permute_215 = torch.ops.aten.permute.default(view_172, [1, 0])
        mm_98 = torch.ops.aten.mm.default(permute_215, view_126);  permute_215 = view_126 = None
        permute_216 = torch.ops.aten.permute.default(mm_98, [1, 0]);  mm_98 = None
        sum_40 = torch.ops.aten.sum.dim_IntList(view_172, [0], True);  view_172 = None
        view_173 = torch.ops.aten.view.default(sum_40, [1152]);  sum_40 = None
        view_174 = torch.ops.aten.view.default(mm_97, [64, 1, 384]);  mm_97 = None
        permute_217 = torch.ops.aten.permute.default(permute_216, [1, 0]);  permute_216 = None
        mul_444 = torch.ops.aten.mul.Tensor(view_174, primals_228);  primals_228 = None
        mul_445 = torch.ops.aten.mul.Tensor(mul_444, 384)
        sum_41 = torch.ops.aten.sum.dim_IntList(mul_444, [2], True)
        mul_446 = torch.ops.aten.mul.Tensor(mul_444, mul_348);  mul_444 = None
        sum_42 = torch.ops.aten.sum.dim_IntList(mul_446, [2], True);  mul_446 = None
        mul_447 = torch.ops.aten.mul.Tensor(mul_348, sum_42);  sum_42 = None
        sub_97 = torch.ops.aten.sub.Tensor(mul_445, sum_41);  mul_445 = sum_41 = None
        sub_98 = torch.ops.aten.sub.Tensor(sub_97, mul_447);  sub_97 = mul_447 = None
        div_23 = torch.ops.aten.div.Tensor(reciprocal_58, 384);  reciprocal_58 = None
        mul_448 = torch.ops.aten.mul.Tensor(div_23, sub_98);  div_23 = sub_98 = None
        mul_449 = torch.ops.aten.mul.Tensor(view_174, mul_348);  mul_348 = None
        sum_43 = torch.ops.aten.sum.dim_IntList(mul_449, [0, 1]);  mul_449 = None
        sum_44 = torch.ops.aten.sum.dim_IntList(view_174, [0, 1]);  view_174 = None
        add_353 = torch.ops.aten.add.Tensor(slice_28, mul_448);  slice_28 = mul_448 = None
        view_175 = torch.ops.aten.view.default(add_353, [64, 384])
        mm_99 = torch.ops.aten.mm.default(view_175, permute_218);  permute_218 = None
        permute_219 = torch.ops.aten.permute.default(view_175, [1, 0])
        mm_100 = torch.ops.aten.mm.default(permute_219, view_124);  permute_219 = view_124 = None
        permute_220 = torch.ops.aten.permute.default(mm_100, [1, 0]);  mm_100 = None
        sum_45 = torch.ops.aten.sum.dim_IntList(view_175, [0], True);  view_175 = None
        view_176 = torch.ops.aten.view.default(sum_45, [384]);  sum_45 = None
        view_177 = torch.ops.aten.view.default(mm_99, [64, 1, 384]);  mm_99 = None
        permute_221 = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
        view_178 = torch.ops.aten.view.default(view_177, [64, 1, 12, 32]);  view_177 = None
        permute_222 = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
        view_179 = torch.ops.aten.view.default(permute_222, [768, 1, 32]);  permute_222 = None
        permute_223 = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
        bmm_40 = torch.ops.aten.bmm.default(permute_223, view_179);  permute_223 = None
        bmm_41 = torch.ops.aten.bmm.default(view_179, permute_224);  view_179 = permute_224 = None
        view_180 = torch.ops.aten.view.default(bmm_40, [64, 12, 197, 32]);  bmm_40 = None
        view_181 = torch.ops.aten.view.default(bmm_41, [64, 12, 1, 197]);  bmm_41 = None
        alias_103 = torch.ops.aten.alias.default(alias_97);  alias_97 = None
        alias_104 = torch.ops.aten.alias.default(alias_103);  alias_103 = None
        mul_450 = torch.ops.aten.mul.Tensor(view_181, alias_104);  view_181 = None
        sum_46 = torch.ops.aten.sum.dim_IntList(mul_450, [-1], True)
        mul_451 = torch.ops.aten.mul.Tensor(alias_104, sum_46);  alias_104 = sum_46 = None
        sub_99 = torch.ops.aten.sub.Tensor(mul_450, mul_451);  mul_450 = mul_451 = None
        view_182 = torch.ops.aten.view.default(sub_99, [768, 1, 197]);  sub_99 = None
        bmm_42 = torch.ops.aten.bmm.default(permute_225, view_182);  permute_225 = None
        bmm_43 = torch.ops.aten.bmm.default(view_182, permute_226);  view_182 = permute_226 = None
        view_183 = torch.ops.aten.view.default(bmm_42, [64, 12, 32, 197]);  bmm_42 = None
        view_184 = torch.ops.aten.view.default(bmm_43, [64, 12, 1, 32]);  bmm_43 = None
        permute_227 = torch.ops.aten.permute.default(view_183, [0, 1, 3, 2]);  view_183 = None
        mul_452 = torch.ops.aten.mul.Tensor(view_184, 0.1767766952966369);  view_184 = None
        view_185 = torch.ops.aten.view.default(mul_452, [64, 1, 384]);  mul_452 = None
        view_186 = torch.ops.aten.view.default(view_185, [64, 384]);  view_185 = None
        permute_228 = torch.ops.aten.permute.default(view_186, [1, 0])
        mm_101 = torch.ops.aten.mm.default(permute_228, view_119);  permute_228 = view_119 = None
        permute_229 = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
        mm_102 = torch.ops.aten.mm.default(view_186, permute_230);  view_186 = permute_230 = None
        view_187 = torch.ops.aten.view.default(mm_102, [64, 1, 384]);  mm_102 = None
        permute_231 = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
        new_zeros_20 = torch.ops.aten.new_zeros.default(view_187, [64, 1, 384], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_12 = torch.ops.aten.slice_scatter.default(new_zeros_20, view_187, 2, 0, 9223372036854775807);  new_zeros_20 = view_187 = None
        new_zeros_21 = torch.ops.aten.new_zeros.default(slice_scatter_12, [64, 197, 384], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_13 = torch.ops.aten.slice_scatter.default(new_zeros_21, slice_scatter_12, 1, 0, 1);  new_zeros_21 = slice_scatter_12 = None
        new_zeros_22 = torch.ops.aten.new_zeros.default(slice_scatter_13, [64, 197, 384], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_14 = torch.ops.aten.slice_scatter.default(new_zeros_22, slice_scatter_13, 0, 0, 9223372036854775807);  new_zeros_22 = slice_scatter_13 = None
        cat_4 = torch.ops.aten.cat.default([permute_227, view_180]);  permute_227 = view_180 = None
        view_188 = torch.ops.aten.view.default(cat_4, [2, 64, 12, 197, 32]);  cat_4 = None
        permute_232 = torch.ops.aten.permute.default(view_188, [1, 3, 0, 2, 4]);  view_188 = None
        clone_118 = torch.ops.aten.clone.default(permute_232, memory_format = torch.contiguous_format);  permute_232 = None
        _unsafe_view_195 = torch.ops.aten._unsafe_view.default(clone_118, [64, 197, 768]);  clone_118 = None
        view_189 = torch.ops.aten.view.default(_unsafe_view_195, [12608, 768]);  _unsafe_view_195 = None
        permute_233 = torch.ops.aten.permute.default(view_189, [1, 0])
        mm_103 = torch.ops.aten.mm.default(permute_233, view_117);  permute_233 = view_117 = None
        permute_234 = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
        mm_104 = torch.ops.aten.mm.default(view_189, permute_235);  view_189 = permute_235 = None
        view_190 = torch.ops.aten.view.default(mm_104, [64, 197, 384]);  mm_104 = None
        add_354 = torch.ops.aten.add.Tensor(slice_scatter_14, view_190);  slice_scatter_14 = view_190 = None
        permute_236 = torch.ops.aten.permute.default(permute_234, [1, 0]);  permute_234 = None
        mul_454 = torch.ops.aten.mul.Tensor(add_354, primals_222);  primals_222 = None
        mul_455 = torch.ops.aten.mul.Tensor(mul_454, 384)
        sum_47 = torch.ops.aten.sum.dim_IntList(mul_454, [2], True)
        mul_456 = torch.ops.aten.mul.Tensor(mul_454, mul_345);  mul_454 = None
        sum_48 = torch.ops.aten.sum.dim_IntList(mul_456, [2], True);  mul_456 = None
        mul_457 = torch.ops.aten.mul.Tensor(mul_345, sum_48);  sum_48 = None
        sub_101 = torch.ops.aten.sub.Tensor(mul_455, sum_47);  mul_455 = sum_47 = None
        sub_102 = torch.ops.aten.sub.Tensor(sub_101, mul_457);  sub_101 = mul_457 = None
        div_24 = torch.ops.aten.div.Tensor(reciprocal_57, 384);  reciprocal_57 = None
        mul_458 = torch.ops.aten.mul.Tensor(div_24, sub_102);  div_24 = sub_102 = None
        mul_459 = torch.ops.aten.mul.Tensor(add_354, mul_345);  mul_345 = None
        sum_49 = torch.ops.aten.sum.dim_IntList(mul_459, [0, 1]);  mul_459 = None
        sum_50 = torch.ops.aten.sum.dim_IntList(add_354, [0, 1]);  add_354 = None
        add_355 = torch.ops.aten.add.Tensor(slice_scatter_11, mul_458);  slice_scatter_11 = mul_458 = None
        new_zeros_23 = torch.ops.aten.new_zeros.default(add_353, [64, 197, 384], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_15 = torch.ops.aten.slice_scatter.default(new_zeros_23, add_353, 1, 0, 1);  new_zeros_23 = add_353 = None
        new_zeros_24 = torch.ops.aten.new_zeros.default(slice_scatter_15, [64, 197, 384], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_16 = torch.ops.aten.slice_scatter.default(new_zeros_24, slice_scatter_15, 0, 0, 9223372036854775807);  new_zeros_24 = slice_scatter_15 = None
        add_356 = torch.ops.aten.add.Tensor(add_355, slice_scatter_16);  add_355 = slice_scatter_16 = None
        slice_30 = torch.ops.aten.slice.Tensor(add_356, 1, 0, 1)
        slice_31 = torch.ops.aten.slice.Tensor(add_356, 1, 1, 197);  add_356 = None
        sum_51 = torch.ops.aten.sum.dim_IntList(slice_30, [0], True);  slice_30 = None
        view_191 = torch.ops.aten.view.default(slice_31, [64, 14, 14, 384]);  slice_31 = None
        sum_52 = torch.ops.aten.sum.dim_IntList(view_191, [0, 1, 2], True)
        view_192 = torch.ops.aten.view.default(sum_52, [384]);  sum_52 = None
        clone_119 = torch.ops.aten.clone.default(view_191, memory_format = torch.contiguous_format)
        _unsafe_view_196 = torch.ops.aten._unsafe_view.default(clone_119, [12544, 384]);  clone_119 = None
        permute_237 = torch.ops.aten.permute.default(_unsafe_view_196, [1, 0])
        mm_105 = torch.ops.aten.mm.default(permute_237, view_115);  permute_237 = view_115 = None
        permute_238 = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
        mm_106 = torch.ops.aten.mm.default(_unsafe_view_196, permute_239);  _unsafe_view_196 = permute_239 = None
        view_193 = torch.ops.aten.view.default(mm_106, [64, 14, 14, 1152]);  mm_106 = None
        permute_240 = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None
        mul_476 = torch.ops.aten.mul.Tensor(view_193, add_363);  view_193 = add_363 = None
        sum_53 = torch.ops.aten.sum.dim_IntList(mul_476, [0, 1, 2], True)
        view_194 = torch.ops.aten.view.default(sum_53, [1152]);  sum_53 = None
        view_195 = torch.ops.aten.view.default(mul_476, [12544, 1152]);  mul_476 = None
        permute_241 = torch.ops.aten.permute.default(view_195, [1, 0])
        mm_107 = torch.ops.aten.mm.default(permute_241, view_114);  permute_241 = view_114 = None
        permute_242 = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
        mm_108 = torch.ops.aten.mm.default(view_195, permute_243);  view_195 = permute_243 = None
        view_196 = torch.ops.aten.view.default(mm_108, [64, 14, 14, 384]);  mm_108 = None
        permute_244 = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
        mul_478 = torch.ops.aten.mul.Tensor(view_196, primals_216);  primals_216 = None
        mul_479 = torch.ops.aten.mul.Tensor(mul_478, 384)
        sum_54 = torch.ops.aten.sum.dim_IntList(mul_478, [3], True)
        mul_480 = torch.ops.aten.mul.Tensor(mul_478, mul_330);  mul_478 = None
        sum_55 = torch.ops.aten.sum.dim_IntList(mul_480, [3], True);  mul_480 = None
        mul_481 = torch.ops.aten.mul.Tensor(mul_330, sum_55);  sum_55 = None
        sub_105 = torch.ops.aten.sub.Tensor(mul_479, sum_54);  mul_479 = sum_54 = None
        sub_106 = torch.ops.aten.sub.Tensor(sub_105, mul_481);  sub_105 = mul_481 = None
        div_25 = torch.ops.aten.div.Tensor(reciprocal_55, 384);  reciprocal_55 = None
        mul_482 = torch.ops.aten.mul.Tensor(div_25, sub_106);  div_25 = sub_106 = None
        mul_483 = torch.ops.aten.mul.Tensor(view_196, mul_330);  mul_330 = None
        sum_56 = torch.ops.aten.sum.dim_IntList(mul_483, [0, 1, 2]);  mul_483 = None
        sum_57 = torch.ops.aten.sum.dim_IntList(view_196, [0, 1, 2]);  view_196 = None
        add_364 = torch.ops.aten.add.Tensor(view_191, mul_482);  view_191 = mul_482 = None
        sum_58 = torch.ops.aten.sum.dim_IntList(add_364, [0, 1, 2], True)
        view_197 = torch.ops.aten.view.default(sum_58, [384]);  sum_58 = None
        view_198 = torch.ops.aten.view.default(add_364, [12544, 384])
        permute_245 = torch.ops.aten.permute.default(view_198, [1, 0])
        mm_109 = torch.ops.aten.mm.default(permute_245, view_113);  permute_245 = view_113 = None
        permute_246 = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
        mm_110 = torch.ops.aten.mm.default(view_198, permute_247);  view_198 = permute_247 = None
        view_199 = torch.ops.aten.view.default(mm_110, [64, 14, 14, 384]);  mm_110 = None
        permute_248 = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
        view_200 = torch.ops.aten.view.default(view_199, [64, 196, 12, 32]);  view_199 = None
        permute_249 = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None
        clone_121 = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
        _unsafe_view_197 = torch.ops.aten._unsafe_view.default(clone_121, [768, 196, 32]);  clone_121 = None
        permute_250 = torch.ops.aten.permute.default(view_112, [0, 2, 1]);  view_112 = None
        bmm_44 = torch.ops.aten.bmm.default(permute_250, _unsafe_view_197);  permute_250 = None
        bmm_45 = torch.ops.aten.bmm.default(_unsafe_view_197, permute_251);  _unsafe_view_197 = permute_251 = None
        view_201 = torch.ops.aten.view.default(bmm_44, [64, 12, 196, 32]);  bmm_44 = None
        view_202 = torch.ops.aten.view.default(bmm_45, [64, 12, 196, 196]);  bmm_45 = None
        alias_105 = torch.ops.aten.alias.default(alias_94);  alias_94 = None
        alias_106 = torch.ops.aten.alias.default(alias_105);  alias_105 = None
        mul_484 = torch.ops.aten.mul.Tensor(view_202, alias_106);  view_202 = None
        sum_59 = torch.ops.aten.sum.dim_IntList(mul_484, [-1], True)
        mul_485 = torch.ops.aten.mul.Tensor(alias_106, sum_59);  alias_106 = sum_59 = None
        sub_107 = torch.ops.aten.sub.Tensor(mul_484, mul_485);  mul_484 = mul_485 = None
        mul_486 = torch.ops.aten.mul.Tensor(sub_107, 0.1767766952966369);  sub_107 = None
        view_203 = torch.ops.aten.view.default(mul_486, [768, 196, 196]);  mul_486 = None
        bmm_46 = torch.ops.aten.bmm.default(permute_252, view_203);  permute_252 = None
        bmm_47 = torch.ops.aten.bmm.default(view_203, permute_253);  view_203 = permute_253 = None
        view_204 = torch.ops.aten.view.default(bmm_46, [64, 12, 32, 196]);  bmm_46 = None
        view_205 = torch.ops.aten.view.default(bmm_47, [64, 12, 196, 32]);  bmm_47 = None
        permute_254 = torch.ops.aten.permute.default(view_204, [0, 1, 3, 2]);  view_204 = None
        cat_5 = torch.ops.aten.cat.default([view_205, permute_254, view_201]);  view_205 = permute_254 = view_201 = None
        view_206 = torch.ops.aten.view.default(cat_5, [3, 64, 12, 196, 32]);  cat_5 = None
        permute_255 = torch.ops.aten.permute.default(view_206, [1, 3, 0, 2, 4]);  view_206 = None
        clone_122 = torch.ops.aten.clone.default(permute_255, memory_format = torch.contiguous_format);  permute_255 = None
        _unsafe_view_198 = torch.ops.aten._unsafe_view.default(clone_122, [64, 14, 14, 1152]);  clone_122 = None
        view_207 = torch.ops.aten.view.default(_unsafe_view_198, [12544, 1152]);  _unsafe_view_198 = None
        permute_256 = torch.ops.aten.permute.default(view_207, [1, 0])
        mm_111 = torch.ops.aten.mm.default(permute_256, view_110);  permute_256 = view_110 = None
        permute_257 = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
        mm_112 = torch.ops.aten.mm.default(view_207, permute_258);  view_207 = permute_258 = None
        view_208 = torch.ops.aten.view.default(mm_112, [64, 14, 14, 384]);  mm_112 = None
        permute_259 = torch.ops.aten.permute.default(permute_257, [1, 0]);  permute_257 = None
        mul_488 = torch.ops.aten.mul.Tensor(view_208, primals_211);  primals_211 = None
        mul_489 = torch.ops.aten.mul.Tensor(mul_488, 384)
        sum_60 = torch.ops.aten.sum.dim_IntList(mul_488, [3], True)
        mul_490 = torch.ops.aten.mul.Tensor(mul_488, mul_327);  mul_488 = None
        sum_61 = torch.ops.aten.sum.dim_IntList(mul_490, [3], True);  mul_490 = None
        mul_491 = torch.ops.aten.mul.Tensor(mul_327, sum_61);  sum_61 = None
        sub_109 = torch.ops.aten.sub.Tensor(mul_489, sum_60);  mul_489 = sum_60 = None
        sub_110 = torch.ops.aten.sub.Tensor(sub_109, mul_491);  sub_109 = mul_491 = None
        div_26 = torch.ops.aten.div.Tensor(reciprocal_54, 384);  reciprocal_54 = None
        mul_492 = torch.ops.aten.mul.Tensor(div_26, sub_110);  div_26 = sub_110 = None
        mul_493 = torch.ops.aten.mul.Tensor(view_208, mul_327);  mul_327 = None
        sum_62 = torch.ops.aten.sum.dim_IntList(mul_493, [0, 1, 2]);  mul_493 = None
        sum_63 = torch.ops.aten.sum.dim_IntList(view_208, [0, 1, 2]);  view_208 = None
        add_365 = torch.ops.aten.add.Tensor(add_364, mul_492);  add_364 = mul_492 = None
        sum_64 = torch.ops.aten.sum.dim_IntList(add_365, [0, 1, 2], True)
        view_209 = torch.ops.aten.view.default(sum_64, [384]);  sum_64 = None
        view_210 = torch.ops.aten.view.default(add_365, [12544, 384])
        permute_260 = torch.ops.aten.permute.default(view_210, [1, 0])
        mm_113 = torch.ops.aten.mm.default(permute_260, view_109);  permute_260 = view_109 = None
        permute_261 = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
        mm_114 = torch.ops.aten.mm.default(view_210, permute_262);  view_210 = permute_262 = None
        view_211 = torch.ops.aten.view.default(mm_114, [64, 14, 14, 1152]);  mm_114 = None
        permute_263 = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None
        mul_510 = torch.ops.aten.mul.Tensor(view_211, add_372);  view_211 = add_372 = None
        sum_65 = torch.ops.aten.sum.dim_IntList(mul_510, [0, 1, 2], True)
        view_212 = torch.ops.aten.view.default(sum_65, [1152]);  sum_65 = None
        view_213 = torch.ops.aten.view.default(mul_510, [12544, 1152]);  mul_510 = None
        permute_264 = torch.ops.aten.permute.default(view_213, [1, 0])
        mm_115 = torch.ops.aten.mm.default(permute_264, view_108);  permute_264 = view_108 = None
        permute_265 = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
        mm_116 = torch.ops.aten.mm.default(view_213, permute_266);  view_213 = permute_266 = None
        view_214 = torch.ops.aten.view.default(mm_116, [64, 14, 14, 384]);  mm_116 = None
        permute_267 = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
        mul_512 = torch.ops.aten.mul.Tensor(view_214, primals_205);  primals_205 = None
        mul_513 = torch.ops.aten.mul.Tensor(mul_512, 384)
        sum_66 = torch.ops.aten.sum.dim_IntList(mul_512, [3], True)
        mul_514 = torch.ops.aten.mul.Tensor(mul_512, mul_312);  mul_512 = None
        sum_67 = torch.ops.aten.sum.dim_IntList(mul_514, [3], True);  mul_514 = None
        mul_515 = torch.ops.aten.mul.Tensor(mul_312, sum_67);  sum_67 = None
        sub_113 = torch.ops.aten.sub.Tensor(mul_513, sum_66);  mul_513 = sum_66 = None
        sub_114 = torch.ops.aten.sub.Tensor(sub_113, mul_515);  sub_113 = mul_515 = None
        div_27 = torch.ops.aten.div.Tensor(reciprocal_52, 384);  reciprocal_52 = None
        mul_516 = torch.ops.aten.mul.Tensor(div_27, sub_114);  div_27 = sub_114 = None
        mul_517 = torch.ops.aten.mul.Tensor(view_214, mul_312);  mul_312 = None
        sum_68 = torch.ops.aten.sum.dim_IntList(mul_517, [0, 1, 2]);  mul_517 = None
        sum_69 = torch.ops.aten.sum.dim_IntList(view_214, [0, 1, 2]);  view_214 = None
        add_373 = torch.ops.aten.add.Tensor(add_365, mul_516);  add_365 = mul_516 = None
        sum_70 = torch.ops.aten.sum.dim_IntList(add_373, [0, 1, 2], True)
        view_215 = torch.ops.aten.view.default(sum_70, [384]);  sum_70 = None
        view_216 = torch.ops.aten.view.default(add_373, [12544, 384])
        permute_268 = torch.ops.aten.permute.default(view_216, [1, 0])
        mm_117 = torch.ops.aten.mm.default(permute_268, view_107);  permute_268 = view_107 = None
        permute_269 = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
        mm_118 = torch.ops.aten.mm.default(view_216, permute_270);  view_216 = permute_270 = None
        view_217 = torch.ops.aten.view.default(mm_118, [64, 14, 14, 384]);  mm_118 = None
        permute_271 = torch.ops.aten.permute.default(permute_269, [1, 0]);  permute_269 = None
        view_218 = torch.ops.aten.view.default(view_217, [64, 196, 12, 32]);  view_217 = None
        permute_272 = torch.ops.aten.permute.default(view_218, [0, 2, 1, 3]);  view_218 = None
        clone_125 = torch.ops.aten.clone.default(permute_272, memory_format = torch.contiguous_format);  permute_272 = None
        _unsafe_view_199 = torch.ops.aten._unsafe_view.default(clone_125, [768, 196, 32]);  clone_125 = None
        permute_273 = torch.ops.aten.permute.default(view_106, [0, 2, 1]);  view_106 = None
        bmm_48 = torch.ops.aten.bmm.default(permute_273, _unsafe_view_199);  permute_273 = None
        bmm_49 = torch.ops.aten.bmm.default(_unsafe_view_199, permute_274);  _unsafe_view_199 = permute_274 = None
        view_219 = torch.ops.aten.view.default(bmm_48, [64, 12, 196, 32]);  bmm_48 = None
        view_220 = torch.ops.aten.view.default(bmm_49, [64, 12, 196, 196]);  bmm_49 = None
        alias_107 = torch.ops.aten.alias.default(alias_91);  alias_91 = None
        alias_108 = torch.ops.aten.alias.default(alias_107);  alias_107 = None
        mul_518 = torch.ops.aten.mul.Tensor(view_220, alias_108);  view_220 = None
        sum_71 = torch.ops.aten.sum.dim_IntList(mul_518, [-1], True)
        mul_519 = torch.ops.aten.mul.Tensor(alias_108, sum_71);  alias_108 = sum_71 = None
        sub_115 = torch.ops.aten.sub.Tensor(mul_518, mul_519);  mul_518 = mul_519 = None
        mul_520 = torch.ops.aten.mul.Tensor(sub_115, 0.1767766952966369);  sub_115 = None
        view_221 = torch.ops.aten.view.default(mul_520, [768, 196, 196]);  mul_520 = None
        bmm_50 = torch.ops.aten.bmm.default(permute_275, view_221);  permute_275 = None
        bmm_51 = torch.ops.aten.bmm.default(view_221, permute_276);  view_221 = permute_276 = None
        view_222 = torch.ops.aten.view.default(bmm_50, [64, 12, 32, 196]);  bmm_50 = None
        view_223 = torch.ops.aten.view.default(bmm_51, [64, 12, 196, 32]);  bmm_51 = None
        permute_277 = torch.ops.aten.permute.default(view_222, [0, 1, 3, 2]);  view_222 = None
        cat_6 = torch.ops.aten.cat.default([view_223, permute_277, view_219]);  view_223 = permute_277 = view_219 = None
        view_224 = torch.ops.aten.view.default(cat_6, [3, 64, 12, 196, 32]);  cat_6 = None
        permute_278 = torch.ops.aten.permute.default(view_224, [1, 3, 0, 2, 4]);  view_224 = None
        clone_126 = torch.ops.aten.clone.default(permute_278, memory_format = torch.contiguous_format);  permute_278 = None
        _unsafe_view_200 = torch.ops.aten._unsafe_view.default(clone_126, [64, 14, 14, 1152]);  clone_126 = None
        view_225 = torch.ops.aten.view.default(_unsafe_view_200, [12544, 1152]);  _unsafe_view_200 = None
        permute_279 = torch.ops.aten.permute.default(view_225, [1, 0])
        mm_119 = torch.ops.aten.mm.default(permute_279, view_104);  permute_279 = view_104 = None
        permute_280 = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
        mm_120 = torch.ops.aten.mm.default(view_225, permute_281);  view_225 = permute_281 = None
        view_226 = torch.ops.aten.view.default(mm_120, [64, 14, 14, 384]);  mm_120 = None
        permute_282 = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
        mul_522 = torch.ops.aten.mul.Tensor(view_226, primals_200);  primals_200 = None
        mul_523 = torch.ops.aten.mul.Tensor(mul_522, 384)
        sum_72 = torch.ops.aten.sum.dim_IntList(mul_522, [3], True)
        mul_524 = torch.ops.aten.mul.Tensor(mul_522, mul_309);  mul_522 = None
        sum_73 = torch.ops.aten.sum.dim_IntList(mul_524, [3], True);  mul_524 = None
        mul_525 = torch.ops.aten.mul.Tensor(mul_309, sum_73);  sum_73 = None
        sub_117 = torch.ops.aten.sub.Tensor(mul_523, sum_72);  mul_523 = sum_72 = None
        sub_118 = torch.ops.aten.sub.Tensor(sub_117, mul_525);  sub_117 = mul_525 = None
        div_28 = torch.ops.aten.div.Tensor(reciprocal_51, 384);  reciprocal_51 = None
        mul_526 = torch.ops.aten.mul.Tensor(div_28, sub_118);  div_28 = sub_118 = None
        mul_527 = torch.ops.aten.mul.Tensor(view_226, mul_309);  mul_309 = None
        sum_74 = torch.ops.aten.sum.dim_IntList(mul_527, [0, 1, 2]);  mul_527 = None
        sum_75 = torch.ops.aten.sum.dim_IntList(view_226, [0, 1, 2]);  view_226 = None
        add_374 = torch.ops.aten.add.Tensor(add_373, mul_526);  add_373 = mul_526 = None
        sum_76 = torch.ops.aten.sum.dim_IntList(add_374, [0, 1, 2], True)
        view_227 = torch.ops.aten.view.default(sum_76, [384]);  sum_76 = None
        view_228 = torch.ops.aten.view.default(add_374, [12544, 384])
        permute_283 = torch.ops.aten.permute.default(view_228, [1, 0])
        mm_121 = torch.ops.aten.mm.default(permute_283, view_103);  permute_283 = view_103 = None
        permute_284 = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
        mm_122 = torch.ops.aten.mm.default(view_228, permute_285);  view_228 = permute_285 = None
        view_229 = torch.ops.aten.view.default(mm_122, [64, 14, 14, 1152]);  mm_122 = None
        permute_286 = torch.ops.aten.permute.default(permute_284, [1, 0]);  permute_284 = None
        mul_544 = torch.ops.aten.mul.Tensor(view_229, add_381);  view_229 = add_381 = None
        sum_77 = torch.ops.aten.sum.dim_IntList(mul_544, [0, 1, 2], True)
        view_230 = torch.ops.aten.view.default(sum_77, [1152]);  sum_77 = None
        view_231 = torch.ops.aten.view.default(mul_544, [12544, 1152]);  mul_544 = None
        permute_287 = torch.ops.aten.permute.default(view_231, [1, 0])
        mm_123 = torch.ops.aten.mm.default(permute_287, view_102);  permute_287 = view_102 = None
        permute_288 = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
        mm_124 = torch.ops.aten.mm.default(view_231, permute_289);  view_231 = permute_289 = None
        view_232 = torch.ops.aten.view.default(mm_124, [64, 14, 14, 384]);  mm_124 = None
        permute_290 = torch.ops.aten.permute.default(permute_288, [1, 0]);  permute_288 = None
        mul_546 = torch.ops.aten.mul.Tensor(view_232, primals_194);  primals_194 = None
        mul_547 = torch.ops.aten.mul.Tensor(mul_546, 384)
        sum_78 = torch.ops.aten.sum.dim_IntList(mul_546, [3], True)
        mul_548 = torch.ops.aten.mul.Tensor(mul_546, mul_294);  mul_546 = None
        sum_79 = torch.ops.aten.sum.dim_IntList(mul_548, [3], True);  mul_548 = None
        mul_549 = torch.ops.aten.mul.Tensor(mul_294, sum_79);  sum_79 = None
        sub_121 = torch.ops.aten.sub.Tensor(mul_547, sum_78);  mul_547 = sum_78 = None
        sub_122 = torch.ops.aten.sub.Tensor(sub_121, mul_549);  sub_121 = mul_549 = None
        div_29 = torch.ops.aten.div.Tensor(reciprocal_49, 384);  reciprocal_49 = None
        mul_550 = torch.ops.aten.mul.Tensor(div_29, sub_122);  div_29 = sub_122 = None
        mul_551 = torch.ops.aten.mul.Tensor(view_232, mul_294);  mul_294 = None
        sum_80 = torch.ops.aten.sum.dim_IntList(mul_551, [0, 1, 2]);  mul_551 = None
        sum_81 = torch.ops.aten.sum.dim_IntList(view_232, [0, 1, 2]);  view_232 = None
        add_382 = torch.ops.aten.add.Tensor(add_374, mul_550);  add_374 = mul_550 = None
        sum_82 = torch.ops.aten.sum.dim_IntList(add_382, [0, 1, 2], True)
        view_233 = torch.ops.aten.view.default(sum_82, [384]);  sum_82 = None
        view_234 = torch.ops.aten.view.default(add_382, [12544, 384])
        permute_291 = torch.ops.aten.permute.default(view_234, [1, 0])
        mm_125 = torch.ops.aten.mm.default(permute_291, view_101);  permute_291 = view_101 = None
        permute_292 = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
        mm_126 = torch.ops.aten.mm.default(view_234, permute_293);  view_234 = permute_293 = None
        view_235 = torch.ops.aten.view.default(mm_126, [64, 14, 14, 384]);  mm_126 = None
        permute_294 = torch.ops.aten.permute.default(permute_292, [1, 0]);  permute_292 = None
        view_236 = torch.ops.aten.view.default(view_235, [64, 196, 12, 32]);  view_235 = None
        permute_295 = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
        clone_129 = torch.ops.aten.clone.default(permute_295, memory_format = torch.contiguous_format);  permute_295 = None
        _unsafe_view_201 = torch.ops.aten._unsafe_view.default(clone_129, [768, 196, 32]);  clone_129 = None
        permute_296 = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
        bmm_52 = torch.ops.aten.bmm.default(permute_296, _unsafe_view_201);  permute_296 = None
        bmm_53 = torch.ops.aten.bmm.default(_unsafe_view_201, permute_297);  _unsafe_view_201 = permute_297 = None
        view_237 = torch.ops.aten.view.default(bmm_52, [64, 12, 196, 32]);  bmm_52 = None
        view_238 = torch.ops.aten.view.default(bmm_53, [64, 12, 196, 196]);  bmm_53 = None
        alias_109 = torch.ops.aten.alias.default(alias_88);  alias_88 = None
        alias_110 = torch.ops.aten.alias.default(alias_109);  alias_109 = None
        mul_552 = torch.ops.aten.mul.Tensor(view_238, alias_110);  view_238 = None
        sum_83 = torch.ops.aten.sum.dim_IntList(mul_552, [-1], True)
        mul_553 = torch.ops.aten.mul.Tensor(alias_110, sum_83);  alias_110 = sum_83 = None
        sub_123 = torch.ops.aten.sub.Tensor(mul_552, mul_553);  mul_552 = mul_553 = None
        mul_554 = torch.ops.aten.mul.Tensor(sub_123, 0.1767766952966369);  sub_123 = None
        view_239 = torch.ops.aten.view.default(mul_554, [768, 196, 196]);  mul_554 = None
        bmm_54 = torch.ops.aten.bmm.default(permute_298, view_239);  permute_298 = None
        bmm_55 = torch.ops.aten.bmm.default(view_239, permute_299);  view_239 = permute_299 = None
        view_240 = torch.ops.aten.view.default(bmm_54, [64, 12, 32, 196]);  bmm_54 = None
        view_241 = torch.ops.aten.view.default(bmm_55, [64, 12, 196, 32]);  bmm_55 = None
        permute_300 = torch.ops.aten.permute.default(view_240, [0, 1, 3, 2]);  view_240 = None
        cat_7 = torch.ops.aten.cat.default([view_241, permute_300, view_237]);  view_241 = permute_300 = view_237 = None
        view_242 = torch.ops.aten.view.default(cat_7, [3, 64, 12, 196, 32]);  cat_7 = None
        permute_301 = torch.ops.aten.permute.default(view_242, [1, 3, 0, 2, 4]);  view_242 = None
        clone_130 = torch.ops.aten.clone.default(permute_301, memory_format = torch.contiguous_format);  permute_301 = None
        _unsafe_view_202 = torch.ops.aten._unsafe_view.default(clone_130, [64, 14, 14, 1152]);  clone_130 = None
        view_243 = torch.ops.aten.view.default(_unsafe_view_202, [12544, 1152]);  _unsafe_view_202 = None
        permute_302 = torch.ops.aten.permute.default(view_243, [1, 0])
        mm_127 = torch.ops.aten.mm.default(permute_302, view_98);  permute_302 = view_98 = None
        permute_303 = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
        mm_128 = torch.ops.aten.mm.default(view_243, permute_304);  view_243 = permute_304 = None
        view_244 = torch.ops.aten.view.default(mm_128, [64, 14, 14, 384]);  mm_128 = None
        permute_305 = torch.ops.aten.permute.default(permute_303, [1, 0]);  permute_303 = None
        mul_556 = torch.ops.aten.mul.Tensor(view_244, primals_189);  primals_189 = None
        mul_557 = torch.ops.aten.mul.Tensor(mul_556, 384)
        sum_84 = torch.ops.aten.sum.dim_IntList(mul_556, [3], True)
        mul_558 = torch.ops.aten.mul.Tensor(mul_556, mul_291);  mul_556 = None
        sum_85 = torch.ops.aten.sum.dim_IntList(mul_558, [3], True);  mul_558 = None
        mul_559 = torch.ops.aten.mul.Tensor(mul_291, sum_85);  sum_85 = None
        sub_125 = torch.ops.aten.sub.Tensor(mul_557, sum_84);  mul_557 = sum_84 = None
        sub_126 = torch.ops.aten.sub.Tensor(sub_125, mul_559);  sub_125 = mul_559 = None
        div_30 = torch.ops.aten.div.Tensor(reciprocal_48, 384);  reciprocal_48 = None
        mul_560 = torch.ops.aten.mul.Tensor(div_30, sub_126);  div_30 = sub_126 = None
        mul_561 = torch.ops.aten.mul.Tensor(view_244, mul_291);  mul_291 = None
        sum_86 = torch.ops.aten.sum.dim_IntList(mul_561, [0, 1, 2]);  mul_561 = None
        sum_87 = torch.ops.aten.sum.dim_IntList(view_244, [0, 1, 2]);  view_244 = None
        add_383 = torch.ops.aten.add.Tensor(add_382, mul_560);  add_382 = mul_560 = None
        sum_88 = torch.ops.aten.sum.dim_IntList(add_383, [0, 1, 2], True)
        view_245 = torch.ops.aten.view.default(sum_88, [384]);  sum_88 = None
        view_246 = torch.ops.aten.view.default(add_383, [12544, 384])
        permute_306 = torch.ops.aten.permute.default(view_246, [1, 0])
        mm_129 = torch.ops.aten.mm.default(permute_306, view_97);  permute_306 = view_97 = None
        permute_307 = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
        mm_130 = torch.ops.aten.mm.default(view_246, permute_308);  view_246 = permute_308 = None
        view_247 = torch.ops.aten.view.default(mm_130, [64, 14, 14, 1152]);  mm_130 = None
        permute_309 = torch.ops.aten.permute.default(permute_307, [1, 0]);  permute_307 = None
        mul_578 = torch.ops.aten.mul.Tensor(view_247, add_390);  view_247 = add_390 = None
        sum_89 = torch.ops.aten.sum.dim_IntList(mul_578, [0, 1, 2], True)
        view_248 = torch.ops.aten.view.default(sum_89, [1152]);  sum_89 = None
        view_249 = torch.ops.aten.view.default(mul_578, [12544, 1152]);  mul_578 = None
        permute_310 = torch.ops.aten.permute.default(view_249, [1, 0])
        mm_131 = torch.ops.aten.mm.default(permute_310, view_96);  permute_310 = view_96 = None
        permute_311 = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
        mm_132 = torch.ops.aten.mm.default(view_249, permute_312);  view_249 = permute_312 = None
        view_250 = torch.ops.aten.view.default(mm_132, [64, 14, 14, 384]);  mm_132 = None
        permute_313 = torch.ops.aten.permute.default(permute_311, [1, 0]);  permute_311 = None
        mul_580 = torch.ops.aten.mul.Tensor(view_250, primals_183);  primals_183 = None
        mul_581 = torch.ops.aten.mul.Tensor(mul_580, 384)
        sum_90 = torch.ops.aten.sum.dim_IntList(mul_580, [3], True)
        mul_582 = torch.ops.aten.mul.Tensor(mul_580, mul_276);  mul_580 = None
        sum_91 = torch.ops.aten.sum.dim_IntList(mul_582, [3], True);  mul_582 = None
        mul_583 = torch.ops.aten.mul.Tensor(mul_276, sum_91);  sum_91 = None
        sub_129 = torch.ops.aten.sub.Tensor(mul_581, sum_90);  mul_581 = sum_90 = None
        sub_130 = torch.ops.aten.sub.Tensor(sub_129, mul_583);  sub_129 = mul_583 = None
        div_31 = torch.ops.aten.div.Tensor(reciprocal_46, 384);  reciprocal_46 = None
        mul_584 = torch.ops.aten.mul.Tensor(div_31, sub_130);  div_31 = sub_130 = None
        mul_585 = torch.ops.aten.mul.Tensor(view_250, mul_276);  mul_276 = None
        sum_92 = torch.ops.aten.sum.dim_IntList(mul_585, [0, 1, 2]);  mul_585 = None
        sum_93 = torch.ops.aten.sum.dim_IntList(view_250, [0, 1, 2]);  view_250 = None
        add_391 = torch.ops.aten.add.Tensor(add_383, mul_584);  add_383 = mul_584 = None
        sum_94 = torch.ops.aten.sum.dim_IntList(add_391, [0, 1, 2], True)
        view_251 = torch.ops.aten.view.default(sum_94, [384]);  sum_94 = None
        view_252 = torch.ops.aten.view.default(add_391, [12544, 384])
        permute_314 = torch.ops.aten.permute.default(view_252, [1, 0])
        mm_133 = torch.ops.aten.mm.default(permute_314, view_95);  permute_314 = view_95 = None
        permute_315 = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
        mm_134 = torch.ops.aten.mm.default(view_252, permute_316);  view_252 = permute_316 = None
        view_253 = torch.ops.aten.view.default(mm_134, [64, 14, 14, 384]);  mm_134 = None
        permute_317 = torch.ops.aten.permute.default(permute_315, [1, 0]);  permute_315 = None
        view_254 = torch.ops.aten.view.default(view_253, [64, 196, 12, 32]);  view_253 = None
        permute_318 = torch.ops.aten.permute.default(view_254, [0, 2, 1, 3]);  view_254 = None
        clone_133 = torch.ops.aten.clone.default(permute_318, memory_format = torch.contiguous_format);  permute_318 = None
        _unsafe_view_203 = torch.ops.aten._unsafe_view.default(clone_133, [768, 196, 32]);  clone_133 = None
        permute_319 = torch.ops.aten.permute.default(view_94, [0, 2, 1]);  view_94 = None
        bmm_56 = torch.ops.aten.bmm.default(permute_319, _unsafe_view_203);  permute_319 = None
        bmm_57 = torch.ops.aten.bmm.default(_unsafe_view_203, permute_320);  _unsafe_view_203 = permute_320 = None
        view_255 = torch.ops.aten.view.default(bmm_56, [64, 12, 196, 32]);  bmm_56 = None
        view_256 = torch.ops.aten.view.default(bmm_57, [64, 12, 196, 196]);  bmm_57 = None
        alias_111 = torch.ops.aten.alias.default(alias_85);  alias_85 = None
        alias_112 = torch.ops.aten.alias.default(alias_111);  alias_111 = None
        mul_586 = torch.ops.aten.mul.Tensor(view_256, alias_112);  view_256 = None
        sum_95 = torch.ops.aten.sum.dim_IntList(mul_586, [-1], True)
        mul_587 = torch.ops.aten.mul.Tensor(alias_112, sum_95);  alias_112 = sum_95 = None
        sub_131 = torch.ops.aten.sub.Tensor(mul_586, mul_587);  mul_586 = mul_587 = None
        mul_588 = torch.ops.aten.mul.Tensor(sub_131, 0.1767766952966369);  sub_131 = None
        view_257 = torch.ops.aten.view.default(mul_588, [768, 196, 196]);  mul_588 = None
        bmm_58 = torch.ops.aten.bmm.default(permute_321, view_257);  permute_321 = None
        bmm_59 = torch.ops.aten.bmm.default(view_257, permute_322);  view_257 = permute_322 = None
        view_258 = torch.ops.aten.view.default(bmm_58, [64, 12, 32, 196]);  bmm_58 = None
        view_259 = torch.ops.aten.view.default(bmm_59, [64, 12, 196, 32]);  bmm_59 = None
        permute_323 = torch.ops.aten.permute.default(view_258, [0, 1, 3, 2]);  view_258 = None
        cat_8 = torch.ops.aten.cat.default([view_259, permute_323, view_255]);  view_259 = permute_323 = view_255 = None
        view_260 = torch.ops.aten.view.default(cat_8, [3, 64, 12, 196, 32]);  cat_8 = None
        permute_324 = torch.ops.aten.permute.default(view_260, [1, 3, 0, 2, 4]);  view_260 = None
        clone_134 = torch.ops.aten.clone.default(permute_324, memory_format = torch.contiguous_format);  permute_324 = None
        _unsafe_view_204 = torch.ops.aten._unsafe_view.default(clone_134, [64, 14, 14, 1152]);  clone_134 = None
        view_261 = torch.ops.aten.view.default(_unsafe_view_204, [12544, 1152]);  _unsafe_view_204 = None
        permute_325 = torch.ops.aten.permute.default(view_261, [1, 0])
        mm_135 = torch.ops.aten.mm.default(permute_325, view_92);  permute_325 = view_92 = None
        permute_326 = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
        mm_136 = torch.ops.aten.mm.default(view_261, permute_327);  view_261 = permute_327 = None
        view_262 = torch.ops.aten.view.default(mm_136, [64, 14, 14, 384]);  mm_136 = None
        permute_328 = torch.ops.aten.permute.default(permute_326, [1, 0]);  permute_326 = None
        mul_590 = torch.ops.aten.mul.Tensor(view_262, primals_178);  primals_178 = None
        mul_591 = torch.ops.aten.mul.Tensor(mul_590, 384)
        sum_96 = torch.ops.aten.sum.dim_IntList(mul_590, [3], True)
        mul_592 = torch.ops.aten.mul.Tensor(mul_590, mul_273);  mul_590 = None
        sum_97 = torch.ops.aten.sum.dim_IntList(mul_592, [3], True);  mul_592 = None
        mul_593 = torch.ops.aten.mul.Tensor(mul_273, sum_97);  sum_97 = None
        sub_133 = torch.ops.aten.sub.Tensor(mul_591, sum_96);  mul_591 = sum_96 = None
        sub_134 = torch.ops.aten.sub.Tensor(sub_133, mul_593);  sub_133 = mul_593 = None
        div_32 = torch.ops.aten.div.Tensor(reciprocal_45, 384);  reciprocal_45 = None
        mul_594 = torch.ops.aten.mul.Tensor(div_32, sub_134);  div_32 = sub_134 = None
        mul_595 = torch.ops.aten.mul.Tensor(view_262, mul_273);  mul_273 = None
        sum_98 = torch.ops.aten.sum.dim_IntList(mul_595, [0, 1, 2]);  mul_595 = None
        sum_99 = torch.ops.aten.sum.dim_IntList(view_262, [0, 1, 2]);  view_262 = None
        add_392 = torch.ops.aten.add.Tensor(add_391, mul_594);  add_391 = mul_594 = None
        sum_100 = torch.ops.aten.sum.dim_IntList(add_392, [0, 1, 2], True)
        view_263 = torch.ops.aten.view.default(sum_100, [384]);  sum_100 = None
        view_264 = torch.ops.aten.view.default(add_392, [12544, 384])
        permute_329 = torch.ops.aten.permute.default(view_264, [1, 0])
        mm_137 = torch.ops.aten.mm.default(permute_329, view_91);  permute_329 = view_91 = None
        permute_330 = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
        mm_138 = torch.ops.aten.mm.default(view_264, permute_331);  view_264 = permute_331 = None
        view_265 = torch.ops.aten.view.default(mm_138, [64, 14, 14, 1152]);  mm_138 = None
        permute_332 = torch.ops.aten.permute.default(permute_330, [1, 0]);  permute_330 = None
        mul_612 = torch.ops.aten.mul.Tensor(view_265, add_399);  view_265 = add_399 = None
        sum_101 = torch.ops.aten.sum.dim_IntList(mul_612, [0, 1, 2], True)
        view_266 = torch.ops.aten.view.default(sum_101, [1152]);  sum_101 = None
        view_267 = torch.ops.aten.view.default(mul_612, [12544, 1152]);  mul_612 = None
        permute_333 = torch.ops.aten.permute.default(view_267, [1, 0])
        mm_139 = torch.ops.aten.mm.default(permute_333, view_90);  permute_333 = view_90 = None
        permute_334 = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
        mm_140 = torch.ops.aten.mm.default(view_267, permute_335);  view_267 = permute_335 = None
        view_268 = torch.ops.aten.view.default(mm_140, [64, 14, 14, 384]);  mm_140 = None
        permute_336 = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
        mul_614 = torch.ops.aten.mul.Tensor(view_268, primals_172);  primals_172 = None
        mul_615 = torch.ops.aten.mul.Tensor(mul_614, 384)
        sum_102 = torch.ops.aten.sum.dim_IntList(mul_614, [3], True)
        mul_616 = torch.ops.aten.mul.Tensor(mul_614, mul_258);  mul_614 = None
        sum_103 = torch.ops.aten.sum.dim_IntList(mul_616, [3], True);  mul_616 = None
        mul_617 = torch.ops.aten.mul.Tensor(mul_258, sum_103);  sum_103 = None
        sub_137 = torch.ops.aten.sub.Tensor(mul_615, sum_102);  mul_615 = sum_102 = None
        sub_138 = torch.ops.aten.sub.Tensor(sub_137, mul_617);  sub_137 = mul_617 = None
        div_33 = torch.ops.aten.div.Tensor(reciprocal_43, 384);  reciprocal_43 = None
        mul_618 = torch.ops.aten.mul.Tensor(div_33, sub_138);  div_33 = sub_138 = None
        mul_619 = torch.ops.aten.mul.Tensor(view_268, mul_258);  mul_258 = None
        sum_104 = torch.ops.aten.sum.dim_IntList(mul_619, [0, 1, 2]);  mul_619 = None
        sum_105 = torch.ops.aten.sum.dim_IntList(view_268, [0, 1, 2]);  view_268 = None
        add_400 = torch.ops.aten.add.Tensor(add_392, mul_618);  add_392 = mul_618 = None
        sum_106 = torch.ops.aten.sum.dim_IntList(add_400, [0, 1, 2], True)
        view_269 = torch.ops.aten.view.default(sum_106, [384]);  sum_106 = None
        view_270 = torch.ops.aten.view.default(add_400, [12544, 384])
        permute_337 = torch.ops.aten.permute.default(view_270, [1, 0])
        mm_141 = torch.ops.aten.mm.default(permute_337, view_89);  permute_337 = view_89 = None
        permute_338 = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
        mm_142 = torch.ops.aten.mm.default(view_270, permute_339);  view_270 = permute_339 = None
        view_271 = torch.ops.aten.view.default(mm_142, [64, 14, 14, 384]);  mm_142 = None
        permute_340 = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
        view_272 = torch.ops.aten.view.default(view_271, [64, 196, 12, 32]);  view_271 = None
        permute_341 = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
        clone_137 = torch.ops.aten.clone.default(permute_341, memory_format = torch.contiguous_format);  permute_341 = None
        _unsafe_view_205 = torch.ops.aten._unsafe_view.default(clone_137, [768, 196, 32]);  clone_137 = None
        permute_342 = torch.ops.aten.permute.default(view_88, [0, 2, 1]);  view_88 = None
        bmm_60 = torch.ops.aten.bmm.default(permute_342, _unsafe_view_205);  permute_342 = None
        bmm_61 = torch.ops.aten.bmm.default(_unsafe_view_205, permute_343);  _unsafe_view_205 = permute_343 = None
        view_273 = torch.ops.aten.view.default(bmm_60, [64, 12, 196, 32]);  bmm_60 = None
        view_274 = torch.ops.aten.view.default(bmm_61, [64, 12, 196, 196]);  bmm_61 = None
        alias_113 = torch.ops.aten.alias.default(alias_82);  alias_82 = None
        alias_114 = torch.ops.aten.alias.default(alias_113);  alias_113 = None
        mul_620 = torch.ops.aten.mul.Tensor(view_274, alias_114);  view_274 = None
        sum_107 = torch.ops.aten.sum.dim_IntList(mul_620, [-1], True)
        mul_621 = torch.ops.aten.mul.Tensor(alias_114, sum_107);  alias_114 = sum_107 = None
        sub_139 = torch.ops.aten.sub.Tensor(mul_620, mul_621);  mul_620 = mul_621 = None
        mul_622 = torch.ops.aten.mul.Tensor(sub_139, 0.1767766952966369);  sub_139 = None
        view_275 = torch.ops.aten.view.default(mul_622, [768, 196, 196]);  mul_622 = None
        bmm_62 = torch.ops.aten.bmm.default(permute_344, view_275);  permute_344 = None
        bmm_63 = torch.ops.aten.bmm.default(view_275, permute_345);  view_275 = permute_345 = None
        view_276 = torch.ops.aten.view.default(bmm_62, [64, 12, 32, 196]);  bmm_62 = None
        view_277 = torch.ops.aten.view.default(bmm_63, [64, 12, 196, 32]);  bmm_63 = None
        permute_346 = torch.ops.aten.permute.default(view_276, [0, 1, 3, 2]);  view_276 = None
        cat_9 = torch.ops.aten.cat.default([view_277, permute_346, view_273]);  view_277 = permute_346 = view_273 = None
        view_278 = torch.ops.aten.view.default(cat_9, [3, 64, 12, 196, 32]);  cat_9 = None
        permute_347 = torch.ops.aten.permute.default(view_278, [1, 3, 0, 2, 4]);  view_278 = None
        clone_138 = torch.ops.aten.clone.default(permute_347, memory_format = torch.contiguous_format);  permute_347 = None
        _unsafe_view_206 = torch.ops.aten._unsafe_view.default(clone_138, [64, 14, 14, 1152]);  clone_138 = None
        view_279 = torch.ops.aten.view.default(_unsafe_view_206, [12544, 1152]);  _unsafe_view_206 = None
        permute_348 = torch.ops.aten.permute.default(view_279, [1, 0])
        mm_143 = torch.ops.aten.mm.default(permute_348, view_86);  permute_348 = view_86 = None
        permute_349 = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
        mm_144 = torch.ops.aten.mm.default(view_279, permute_350);  view_279 = permute_350 = None
        view_280 = torch.ops.aten.view.default(mm_144, [64, 14, 14, 384]);  mm_144 = None
        permute_351 = torch.ops.aten.permute.default(permute_349, [1, 0]);  permute_349 = None
        mul_624 = torch.ops.aten.mul.Tensor(view_280, primals_167);  primals_167 = None
        mul_625 = torch.ops.aten.mul.Tensor(mul_624, 384)
        sum_108 = torch.ops.aten.sum.dim_IntList(mul_624, [3], True)
        mul_626 = torch.ops.aten.mul.Tensor(mul_624, mul_255);  mul_624 = None
        sum_109 = torch.ops.aten.sum.dim_IntList(mul_626, [3], True);  mul_626 = None
        mul_627 = torch.ops.aten.mul.Tensor(mul_255, sum_109);  sum_109 = None
        sub_141 = torch.ops.aten.sub.Tensor(mul_625, sum_108);  mul_625 = sum_108 = None
        sub_142 = torch.ops.aten.sub.Tensor(sub_141, mul_627);  sub_141 = mul_627 = None
        div_34 = torch.ops.aten.div.Tensor(reciprocal_42, 384);  reciprocal_42 = None
        mul_628 = torch.ops.aten.mul.Tensor(div_34, sub_142);  div_34 = sub_142 = None
        mul_629 = torch.ops.aten.mul.Tensor(view_280, mul_255);  mul_255 = None
        sum_110 = torch.ops.aten.sum.dim_IntList(mul_629, [0, 1, 2]);  mul_629 = None
        sum_111 = torch.ops.aten.sum.dim_IntList(view_280, [0, 1, 2]);  view_280 = None
        add_401 = torch.ops.aten.add.Tensor(add_400, mul_628);  add_400 = mul_628 = None
        sum_112 = torch.ops.aten.sum.dim_IntList(add_401, [0, 1, 2], True)
        view_281 = torch.ops.aten.view.default(sum_112, [384]);  sum_112 = None
        view_282 = torch.ops.aten.view.default(add_401, [12544, 384])
        permute_352 = torch.ops.aten.permute.default(view_282, [1, 0])
        mm_145 = torch.ops.aten.mm.default(permute_352, view_85);  permute_352 = view_85 = None
        permute_353 = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
        mm_146 = torch.ops.aten.mm.default(view_282, permute_354);  view_282 = permute_354 = None
        view_283 = torch.ops.aten.view.default(mm_146, [64, 14, 14, 1152]);  mm_146 = None
        permute_355 = torch.ops.aten.permute.default(permute_353, [1, 0]);  permute_353 = None
        mul_646 = torch.ops.aten.mul.Tensor(view_283, add_408);  view_283 = add_408 = None
        sum_113 = torch.ops.aten.sum.dim_IntList(mul_646, [0, 1, 2], True)
        view_284 = torch.ops.aten.view.default(sum_113, [1152]);  sum_113 = None
        view_285 = torch.ops.aten.view.default(mul_646, [12544, 1152]);  mul_646 = None
        permute_356 = torch.ops.aten.permute.default(view_285, [1, 0])
        mm_147 = torch.ops.aten.mm.default(permute_356, view_84);  permute_356 = view_84 = None
        permute_357 = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
        mm_148 = torch.ops.aten.mm.default(view_285, permute_358);  view_285 = permute_358 = None
        view_286 = torch.ops.aten.view.default(mm_148, [64, 14, 14, 384]);  mm_148 = None
        permute_359 = torch.ops.aten.permute.default(permute_357, [1, 0]);  permute_357 = None
        mul_648 = torch.ops.aten.mul.Tensor(view_286, primals_161);  primals_161 = None
        mul_649 = torch.ops.aten.mul.Tensor(mul_648, 384)
        sum_114 = torch.ops.aten.sum.dim_IntList(mul_648, [3], True)
        mul_650 = torch.ops.aten.mul.Tensor(mul_648, mul_240);  mul_648 = None
        sum_115 = torch.ops.aten.sum.dim_IntList(mul_650, [3], True);  mul_650 = None
        mul_651 = torch.ops.aten.mul.Tensor(mul_240, sum_115);  sum_115 = None
        sub_145 = torch.ops.aten.sub.Tensor(mul_649, sum_114);  mul_649 = sum_114 = None
        sub_146 = torch.ops.aten.sub.Tensor(sub_145, mul_651);  sub_145 = mul_651 = None
        div_35 = torch.ops.aten.div.Tensor(reciprocal_40, 384);  reciprocal_40 = None
        mul_652 = torch.ops.aten.mul.Tensor(div_35, sub_146);  div_35 = sub_146 = None
        mul_653 = torch.ops.aten.mul.Tensor(view_286, mul_240);  mul_240 = None
        sum_116 = torch.ops.aten.sum.dim_IntList(mul_653, [0, 1, 2]);  mul_653 = None
        sum_117 = torch.ops.aten.sum.dim_IntList(view_286, [0, 1, 2]);  view_286 = None
        add_409 = torch.ops.aten.add.Tensor(add_401, mul_652);  add_401 = mul_652 = None
        sum_118 = torch.ops.aten.sum.dim_IntList(add_409, [0, 1, 2], True)
        view_287 = torch.ops.aten.view.default(sum_118, [384]);  sum_118 = None
        view_288 = torch.ops.aten.view.default(add_409, [12544, 384])
        permute_360 = torch.ops.aten.permute.default(view_288, [1, 0])
        mm_149 = torch.ops.aten.mm.default(permute_360, view_83);  permute_360 = view_83 = None
        permute_361 = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
        mm_150 = torch.ops.aten.mm.default(view_288, permute_362);  view_288 = permute_362 = None
        view_289 = torch.ops.aten.view.default(mm_150, [64, 14, 14, 384]);  mm_150 = None
        permute_363 = torch.ops.aten.permute.default(permute_361, [1, 0]);  permute_361 = None
        view_290 = torch.ops.aten.view.default(view_289, [64, 196, 12, 32]);  view_289 = None
        permute_364 = torch.ops.aten.permute.default(view_290, [0, 2, 1, 3]);  view_290 = None
        clone_141 = torch.ops.aten.clone.default(permute_364, memory_format = torch.contiguous_format);  permute_364 = None
        _unsafe_view_207 = torch.ops.aten._unsafe_view.default(clone_141, [768, 196, 32]);  clone_141 = None
        permute_365 = torch.ops.aten.permute.default(view_82, [0, 2, 1]);  view_82 = None
        bmm_64 = torch.ops.aten.bmm.default(permute_365, _unsafe_view_207);  permute_365 = None
        bmm_65 = torch.ops.aten.bmm.default(_unsafe_view_207, permute_366);  _unsafe_view_207 = permute_366 = None
        view_291 = torch.ops.aten.view.default(bmm_64, [64, 12, 196, 32]);  bmm_64 = None
        view_292 = torch.ops.aten.view.default(bmm_65, [64, 12, 196, 196]);  bmm_65 = None
        alias_115 = torch.ops.aten.alias.default(alias_79);  alias_79 = None
        alias_116 = torch.ops.aten.alias.default(alias_115);  alias_115 = None
        mul_654 = torch.ops.aten.mul.Tensor(view_292, alias_116);  view_292 = None
        sum_119 = torch.ops.aten.sum.dim_IntList(mul_654, [-1], True)
        mul_655 = torch.ops.aten.mul.Tensor(alias_116, sum_119);  alias_116 = sum_119 = None
        sub_147 = torch.ops.aten.sub.Tensor(mul_654, mul_655);  mul_654 = mul_655 = None
        mul_656 = torch.ops.aten.mul.Tensor(sub_147, 0.1767766952966369);  sub_147 = None
        view_293 = torch.ops.aten.view.default(mul_656, [768, 196, 196]);  mul_656 = None
        bmm_66 = torch.ops.aten.bmm.default(permute_367, view_293);  permute_367 = None
        bmm_67 = torch.ops.aten.bmm.default(view_293, permute_368);  view_293 = permute_368 = None
        view_294 = torch.ops.aten.view.default(bmm_66, [64, 12, 32, 196]);  bmm_66 = None
        view_295 = torch.ops.aten.view.default(bmm_67, [64, 12, 196, 32]);  bmm_67 = None
        permute_369 = torch.ops.aten.permute.default(view_294, [0, 1, 3, 2]);  view_294 = None
        cat_10 = torch.ops.aten.cat.default([view_295, permute_369, view_291]);  view_295 = permute_369 = view_291 = None
        view_296 = torch.ops.aten.view.default(cat_10, [3, 64, 12, 196, 32]);  cat_10 = None
        permute_370 = torch.ops.aten.permute.default(view_296, [1, 3, 0, 2, 4]);  view_296 = None
        clone_142 = torch.ops.aten.clone.default(permute_370, memory_format = torch.contiguous_format);  permute_370 = None
        _unsafe_view_208 = torch.ops.aten._unsafe_view.default(clone_142, [64, 14, 14, 1152]);  clone_142 = None
        view_297 = torch.ops.aten.view.default(_unsafe_view_208, [12544, 1152]);  _unsafe_view_208 = None
        permute_371 = torch.ops.aten.permute.default(view_297, [1, 0])
        mm_151 = torch.ops.aten.mm.default(permute_371, view_80);  permute_371 = view_80 = None
        permute_372 = torch.ops.aten.permute.default(mm_151, [1, 0]);  mm_151 = None
        mm_152 = torch.ops.aten.mm.default(view_297, permute_373);  view_297 = permute_373 = None
        view_298 = torch.ops.aten.view.default(mm_152, [64, 14, 14, 384]);  mm_152 = None
        permute_374 = torch.ops.aten.permute.default(permute_372, [1, 0]);  permute_372 = None
        mul_658 = torch.ops.aten.mul.Tensor(view_298, primals_156);  primals_156 = None
        mul_659 = torch.ops.aten.mul.Tensor(mul_658, 384)
        sum_120 = torch.ops.aten.sum.dim_IntList(mul_658, [3], True)
        mul_660 = torch.ops.aten.mul.Tensor(mul_658, mul_237);  mul_658 = None
        sum_121 = torch.ops.aten.sum.dim_IntList(mul_660, [3], True);  mul_660 = None
        mul_661 = torch.ops.aten.mul.Tensor(mul_237, sum_121);  sum_121 = None
        sub_149 = torch.ops.aten.sub.Tensor(mul_659, sum_120);  mul_659 = sum_120 = None
        sub_150 = torch.ops.aten.sub.Tensor(sub_149, mul_661);  sub_149 = mul_661 = None
        div_36 = torch.ops.aten.div.Tensor(reciprocal_39, 384);  reciprocal_39 = None
        mul_662 = torch.ops.aten.mul.Tensor(div_36, sub_150);  div_36 = sub_150 = None
        mul_663 = torch.ops.aten.mul.Tensor(view_298, mul_237);  mul_237 = None
        sum_122 = torch.ops.aten.sum.dim_IntList(mul_663, [0, 1, 2]);  mul_663 = None
        sum_123 = torch.ops.aten.sum.dim_IntList(view_298, [0, 1, 2]);  view_298 = None
        add_410 = torch.ops.aten.add.Tensor(add_409, mul_662);  add_409 = mul_662 = None
        sum_124 = torch.ops.aten.sum.dim_IntList(add_410, [0, 1, 2], True)
        view_299 = torch.ops.aten.view.default(sum_124, [384]);  sum_124 = None
        view_300 = torch.ops.aten.view.default(add_410, [12544, 384])
        permute_375 = torch.ops.aten.permute.default(view_300, [1, 0])
        mm_153 = torch.ops.aten.mm.default(permute_375, view_79);  permute_375 = view_79 = None
        permute_376 = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
        mm_154 = torch.ops.aten.mm.default(view_300, permute_377);  view_300 = permute_377 = None
        view_301 = torch.ops.aten.view.default(mm_154, [64, 14, 14, 1152]);  mm_154 = None
        permute_378 = torch.ops.aten.permute.default(permute_376, [1, 0]);  permute_376 = None
        mul_680 = torch.ops.aten.mul.Tensor(view_301, add_417);  view_301 = add_417 = None
        sum_125 = torch.ops.aten.sum.dim_IntList(mul_680, [0, 1, 2], True)
        view_302 = torch.ops.aten.view.default(sum_125, [1152]);  sum_125 = None
        view_303 = torch.ops.aten.view.default(mul_680, [12544, 1152]);  mul_680 = None
        permute_379 = torch.ops.aten.permute.default(view_303, [1, 0])
        mm_155 = torch.ops.aten.mm.default(permute_379, view_78);  permute_379 = view_78 = None
        permute_380 = torch.ops.aten.permute.default(mm_155, [1, 0]);  mm_155 = None
        mm_156 = torch.ops.aten.mm.default(view_303, permute_381);  view_303 = permute_381 = None
        view_304 = torch.ops.aten.view.default(mm_156, [64, 14, 14, 384]);  mm_156 = None
        permute_382 = torch.ops.aten.permute.default(permute_380, [1, 0]);  permute_380 = None
        mul_682 = torch.ops.aten.mul.Tensor(view_304, primals_150);  primals_150 = None
        mul_683 = torch.ops.aten.mul.Tensor(mul_682, 384)
        sum_126 = torch.ops.aten.sum.dim_IntList(mul_682, [3], True)
        mul_684 = torch.ops.aten.mul.Tensor(mul_682, mul_222);  mul_682 = None
        sum_127 = torch.ops.aten.sum.dim_IntList(mul_684, [3], True);  mul_684 = None
        mul_685 = torch.ops.aten.mul.Tensor(mul_222, sum_127);  sum_127 = None
        sub_153 = torch.ops.aten.sub.Tensor(mul_683, sum_126);  mul_683 = sum_126 = None
        sub_154 = torch.ops.aten.sub.Tensor(sub_153, mul_685);  sub_153 = mul_685 = None
        div_37 = torch.ops.aten.div.Tensor(reciprocal_37, 384);  reciprocal_37 = None
        mul_686 = torch.ops.aten.mul.Tensor(div_37, sub_154);  div_37 = sub_154 = None
        mul_687 = torch.ops.aten.mul.Tensor(view_304, mul_222);  mul_222 = None
        sum_128 = torch.ops.aten.sum.dim_IntList(mul_687, [0, 1, 2]);  mul_687 = None
        sum_129 = torch.ops.aten.sum.dim_IntList(view_304, [0, 1, 2]);  view_304 = None
        add_418 = torch.ops.aten.add.Tensor(add_410, mul_686);  add_410 = mul_686 = None
        sum_130 = torch.ops.aten.sum.dim_IntList(add_418, [0, 1, 2], True)
        view_305 = torch.ops.aten.view.default(sum_130, [384]);  sum_130 = None
        view_306 = torch.ops.aten.view.default(add_418, [12544, 384])
        permute_383 = torch.ops.aten.permute.default(view_306, [1, 0])
        mm_157 = torch.ops.aten.mm.default(permute_383, view_77);  permute_383 = view_77 = None
        permute_384 = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
        mm_158 = torch.ops.aten.mm.default(view_306, permute_385);  view_306 = permute_385 = None
        view_307 = torch.ops.aten.view.default(mm_158, [64, 14, 14, 384]);  mm_158 = None
        permute_386 = torch.ops.aten.permute.default(permute_384, [1, 0]);  permute_384 = None
        view_308 = torch.ops.aten.view.default(view_307, [64, 196, 12, 32]);  view_307 = None
        permute_387 = torch.ops.aten.permute.default(view_308, [0, 2, 1, 3]);  view_308 = None
        clone_145 = torch.ops.aten.clone.default(permute_387, memory_format = torch.contiguous_format);  permute_387 = None
        _unsafe_view_209 = torch.ops.aten._unsafe_view.default(clone_145, [768, 196, 32]);  clone_145 = None
        permute_388 = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None
        bmm_68 = torch.ops.aten.bmm.default(permute_388, _unsafe_view_209);  permute_388 = None
        bmm_69 = torch.ops.aten.bmm.default(_unsafe_view_209, permute_389);  _unsafe_view_209 = permute_389 = None
        view_309 = torch.ops.aten.view.default(bmm_68, [64, 12, 196, 32]);  bmm_68 = None
        view_310 = torch.ops.aten.view.default(bmm_69, [64, 12, 196, 196]);  bmm_69 = None
        alias_117 = torch.ops.aten.alias.default(alias_76);  alias_76 = None
        alias_118 = torch.ops.aten.alias.default(alias_117);  alias_117 = None
        mul_688 = torch.ops.aten.mul.Tensor(view_310, alias_118);  view_310 = None
        sum_131 = torch.ops.aten.sum.dim_IntList(mul_688, [-1], True)
        mul_689 = torch.ops.aten.mul.Tensor(alias_118, sum_131);  alias_118 = sum_131 = None
        sub_155 = torch.ops.aten.sub.Tensor(mul_688, mul_689);  mul_688 = mul_689 = None
        mul_690 = torch.ops.aten.mul.Tensor(sub_155, 0.1767766952966369);  sub_155 = None
        view_311 = torch.ops.aten.view.default(mul_690, [768, 196, 196]);  mul_690 = None
        bmm_70 = torch.ops.aten.bmm.default(permute_390, view_311);  permute_390 = None
        bmm_71 = torch.ops.aten.bmm.default(view_311, permute_391);  view_311 = permute_391 = None
        view_312 = torch.ops.aten.view.default(bmm_70, [64, 12, 32, 196]);  bmm_70 = None
        view_313 = torch.ops.aten.view.default(bmm_71, [64, 12, 196, 32]);  bmm_71 = None
        permute_392 = torch.ops.aten.permute.default(view_312, [0, 1, 3, 2]);  view_312 = None
        cat_11 = torch.ops.aten.cat.default([view_313, permute_392, view_309]);  view_313 = permute_392 = view_309 = None
        view_314 = torch.ops.aten.view.default(cat_11, [3, 64, 12, 196, 32]);  cat_11 = None
        permute_393 = torch.ops.aten.permute.default(view_314, [1, 3, 0, 2, 4]);  view_314 = None
        clone_146 = torch.ops.aten.clone.default(permute_393, memory_format = torch.contiguous_format);  permute_393 = None
        _unsafe_view_210 = torch.ops.aten._unsafe_view.default(clone_146, [64, 14, 14, 1152]);  clone_146 = None
        view_315 = torch.ops.aten.view.default(_unsafe_view_210, [12544, 1152]);  _unsafe_view_210 = None
        permute_394 = torch.ops.aten.permute.default(view_315, [1, 0])
        mm_159 = torch.ops.aten.mm.default(permute_394, view_74);  permute_394 = view_74 = None
        permute_395 = torch.ops.aten.permute.default(mm_159, [1, 0]);  mm_159 = None
        mm_160 = torch.ops.aten.mm.default(view_315, permute_396);  view_315 = permute_396 = None
        view_316 = torch.ops.aten.view.default(mm_160, [64, 14, 14, 384]);  mm_160 = None
        permute_397 = torch.ops.aten.permute.default(permute_395, [1, 0]);  permute_395 = None
        mul_692 = torch.ops.aten.mul.Tensor(view_316, primals_145);  primals_145 = None
        mul_693 = torch.ops.aten.mul.Tensor(mul_692, 384)
        sum_132 = torch.ops.aten.sum.dim_IntList(mul_692, [3], True)
        mul_694 = torch.ops.aten.mul.Tensor(mul_692, mul_219);  mul_692 = None
        sum_133 = torch.ops.aten.sum.dim_IntList(mul_694, [3], True);  mul_694 = None
        mul_695 = torch.ops.aten.mul.Tensor(mul_219, sum_133);  sum_133 = None
        sub_157 = torch.ops.aten.sub.Tensor(mul_693, sum_132);  mul_693 = sum_132 = None
        sub_158 = torch.ops.aten.sub.Tensor(sub_157, mul_695);  sub_157 = mul_695 = None
        div_38 = torch.ops.aten.div.Tensor(reciprocal_36, 384);  reciprocal_36 = None
        mul_696 = torch.ops.aten.mul.Tensor(div_38, sub_158);  div_38 = sub_158 = None
        mul_697 = torch.ops.aten.mul.Tensor(view_316, mul_219);  mul_219 = None
        sum_134 = torch.ops.aten.sum.dim_IntList(mul_697, [0, 1, 2]);  mul_697 = None
        sum_135 = torch.ops.aten.sum.dim_IntList(view_316, [0, 1, 2]);  view_316 = None
        add_419 = torch.ops.aten.add.Tensor(add_418, mul_696);  add_418 = mul_696 = None
        sum_136 = torch.ops.aten.sum.dim_IntList(add_419, [0, 1, 2], True)
        view_317 = torch.ops.aten.view.default(sum_136, [384]);  sum_136 = None
        view_318 = torch.ops.aten.view.default(add_419, [12544, 384])
        permute_398 = torch.ops.aten.permute.default(view_318, [1, 0])
        mm_161 = torch.ops.aten.mm.default(permute_398, view_73);  permute_398 = view_73 = None
        permute_399 = torch.ops.aten.permute.default(mm_161, [1, 0]);  mm_161 = None
        mm_162 = torch.ops.aten.mm.default(view_318, permute_400);  view_318 = permute_400 = None
        view_319 = torch.ops.aten.view.default(mm_162, [64, 14, 14, 1152]);  mm_162 = None
        permute_401 = torch.ops.aten.permute.default(permute_399, [1, 0]);  permute_399 = None
        mul_714 = torch.ops.aten.mul.Tensor(view_319, add_426);  view_319 = add_426 = None
        sum_137 = torch.ops.aten.sum.dim_IntList(mul_714, [0, 1, 2], True)
        view_320 = torch.ops.aten.view.default(sum_137, [1152]);  sum_137 = None
        view_321 = torch.ops.aten.view.default(mul_714, [12544, 1152]);  mul_714 = None
        permute_402 = torch.ops.aten.permute.default(view_321, [1, 0])
        mm_163 = torch.ops.aten.mm.default(permute_402, view_72);  permute_402 = view_72 = None
        permute_403 = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
        mm_164 = torch.ops.aten.mm.default(view_321, permute_404);  view_321 = permute_404 = None
        view_322 = torch.ops.aten.view.default(mm_164, [64, 14, 14, 384]);  mm_164 = None
        permute_405 = torch.ops.aten.permute.default(permute_403, [1, 0]);  permute_403 = None
        mul_716 = torch.ops.aten.mul.Tensor(view_322, primals_139);  primals_139 = None
        mul_717 = torch.ops.aten.mul.Tensor(mul_716, 384)
        sum_138 = torch.ops.aten.sum.dim_IntList(mul_716, [3], True)
        mul_718 = torch.ops.aten.mul.Tensor(mul_716, mul_204);  mul_716 = None
        sum_139 = torch.ops.aten.sum.dim_IntList(mul_718, [3], True);  mul_718 = None
        mul_719 = torch.ops.aten.mul.Tensor(mul_204, sum_139);  sum_139 = None
        sub_161 = torch.ops.aten.sub.Tensor(mul_717, sum_138);  mul_717 = sum_138 = None
        sub_162 = torch.ops.aten.sub.Tensor(sub_161, mul_719);  sub_161 = mul_719 = None
        div_39 = torch.ops.aten.div.Tensor(reciprocal_34, 384);  reciprocal_34 = None
        mul_720 = torch.ops.aten.mul.Tensor(div_39, sub_162);  div_39 = sub_162 = None
        mul_721 = torch.ops.aten.mul.Tensor(view_322, mul_204);  mul_204 = None
        sum_140 = torch.ops.aten.sum.dim_IntList(mul_721, [0, 1, 2]);  mul_721 = None
        sum_141 = torch.ops.aten.sum.dim_IntList(view_322, [0, 1, 2]);  view_322 = None
        add_427 = torch.ops.aten.add.Tensor(add_419, mul_720);  add_419 = mul_720 = None
        sum_142 = torch.ops.aten.sum.dim_IntList(add_427, [0, 1, 2], True)
        view_323 = torch.ops.aten.view.default(sum_142, [384]);  sum_142 = None
        view_324 = torch.ops.aten.view.default(add_427, [12544, 384])
        permute_406 = torch.ops.aten.permute.default(view_324, [1, 0])
        mm_165 = torch.ops.aten.mm.default(permute_406, view_71);  permute_406 = view_71 = None
        permute_407 = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
        mm_166 = torch.ops.aten.mm.default(view_324, permute_408);  view_324 = permute_408 = None
        view_325 = torch.ops.aten.view.default(mm_166, [64, 14, 14, 384]);  mm_166 = None
        permute_409 = torch.ops.aten.permute.default(permute_407, [1, 0]);  permute_407 = None
        view_326 = torch.ops.aten.view.default(view_325, [64, 196, 12, 32]);  view_325 = None
        permute_410 = torch.ops.aten.permute.default(view_326, [0, 2, 1, 3]);  view_326 = None
        clone_149 = torch.ops.aten.clone.default(permute_410, memory_format = torch.contiguous_format);  permute_410 = None
        _unsafe_view_211 = torch.ops.aten._unsafe_view.default(clone_149, [768, 196, 32]);  clone_149 = None
        permute_411 = torch.ops.aten.permute.default(view_70, [0, 2, 1]);  view_70 = None
        bmm_72 = torch.ops.aten.bmm.default(permute_411, _unsafe_view_211);  permute_411 = None
        bmm_73 = torch.ops.aten.bmm.default(_unsafe_view_211, permute_412);  _unsafe_view_211 = permute_412 = None
        view_327 = torch.ops.aten.view.default(bmm_72, [64, 12, 196, 32]);  bmm_72 = None
        view_328 = torch.ops.aten.view.default(bmm_73, [64, 12, 196, 196]);  bmm_73 = None
        alias_119 = torch.ops.aten.alias.default(alias_73);  alias_73 = None
        alias_120 = torch.ops.aten.alias.default(alias_119);  alias_119 = None
        mul_722 = torch.ops.aten.mul.Tensor(view_328, alias_120);  view_328 = None
        sum_143 = torch.ops.aten.sum.dim_IntList(mul_722, [-1], True)
        mul_723 = torch.ops.aten.mul.Tensor(alias_120, sum_143);  alias_120 = sum_143 = None
        sub_163 = torch.ops.aten.sub.Tensor(mul_722, mul_723);  mul_722 = mul_723 = None
        mul_724 = torch.ops.aten.mul.Tensor(sub_163, 0.1767766952966369);  sub_163 = None
        view_329 = torch.ops.aten.view.default(mul_724, [768, 196, 196]);  mul_724 = None
        bmm_74 = torch.ops.aten.bmm.default(permute_413, view_329);  permute_413 = None
        bmm_75 = torch.ops.aten.bmm.default(view_329, permute_414);  view_329 = permute_414 = None
        view_330 = torch.ops.aten.view.default(bmm_74, [64, 12, 32, 196]);  bmm_74 = None
        view_331 = torch.ops.aten.view.default(bmm_75, [64, 12, 196, 32]);  bmm_75 = None
        permute_415 = torch.ops.aten.permute.default(view_330, [0, 1, 3, 2]);  view_330 = None
        cat_12 = torch.ops.aten.cat.default([view_331, permute_415, view_327]);  view_331 = permute_415 = view_327 = None
        view_332 = torch.ops.aten.view.default(cat_12, [3, 64, 12, 196, 32]);  cat_12 = None
        permute_416 = torch.ops.aten.permute.default(view_332, [1, 3, 0, 2, 4]);  view_332 = None
        clone_150 = torch.ops.aten.clone.default(permute_416, memory_format = torch.contiguous_format);  permute_416 = None
        _unsafe_view_212 = torch.ops.aten._unsafe_view.default(clone_150, [64, 14, 14, 1152]);  clone_150 = None
        view_333 = torch.ops.aten.view.default(_unsafe_view_212, [12544, 1152]);  _unsafe_view_212 = None
        permute_417 = torch.ops.aten.permute.default(view_333, [1, 0])
        mm_167 = torch.ops.aten.mm.default(permute_417, view_68);  permute_417 = view_68 = None
        permute_418 = torch.ops.aten.permute.default(mm_167, [1, 0]);  mm_167 = None
        mm_168 = torch.ops.aten.mm.default(view_333, permute_419);  view_333 = permute_419 = None
        view_334 = torch.ops.aten.view.default(mm_168, [64, 14, 14, 384]);  mm_168 = None
        permute_420 = torch.ops.aten.permute.default(permute_418, [1, 0]);  permute_418 = None
        mul_726 = torch.ops.aten.mul.Tensor(view_334, primals_134);  primals_134 = None
        mul_727 = torch.ops.aten.mul.Tensor(mul_726, 384)
        sum_144 = torch.ops.aten.sum.dim_IntList(mul_726, [3], True)
        mul_728 = torch.ops.aten.mul.Tensor(mul_726, mul_201);  mul_726 = None
        sum_145 = torch.ops.aten.sum.dim_IntList(mul_728, [3], True);  mul_728 = None
        mul_729 = torch.ops.aten.mul.Tensor(mul_201, sum_145);  sum_145 = None
        sub_165 = torch.ops.aten.sub.Tensor(mul_727, sum_144);  mul_727 = sum_144 = None
        sub_166 = torch.ops.aten.sub.Tensor(sub_165, mul_729);  sub_165 = mul_729 = None
        div_40 = torch.ops.aten.div.Tensor(reciprocal_33, 384);  reciprocal_33 = None
        mul_730 = torch.ops.aten.mul.Tensor(div_40, sub_166);  div_40 = sub_166 = None
        mul_731 = torch.ops.aten.mul.Tensor(view_334, mul_201);  mul_201 = None
        sum_146 = torch.ops.aten.sum.dim_IntList(mul_731, [0, 1, 2]);  mul_731 = None
        sum_147 = torch.ops.aten.sum.dim_IntList(view_334, [0, 1, 2]);  view_334 = None
        add_428 = torch.ops.aten.add.Tensor(add_427, mul_730);  add_427 = mul_730 = None
        sum_148 = torch.ops.aten.sum.dim_IntList(add_428, [0, 1, 2], True)
        view_335 = torch.ops.aten.view.default(sum_148, [384]);  sum_148 = None
        view_336 = torch.ops.aten.view.default(add_428, [12544, 384])
        permute_421 = torch.ops.aten.permute.default(view_336, [1, 0])
        mm_169 = torch.ops.aten.mm.default(permute_421, view_67);  permute_421 = view_67 = None
        permute_422 = torch.ops.aten.permute.default(mm_169, [1, 0]);  mm_169 = None
        mm_170 = torch.ops.aten.mm.default(view_336, permute_423);  view_336 = permute_423 = None
        view_337 = torch.ops.aten.view.default(mm_170, [64, 14, 14, 1152]);  mm_170 = None
        permute_424 = torch.ops.aten.permute.default(permute_422, [1, 0]);  permute_422 = None
        mul_748 = torch.ops.aten.mul.Tensor(view_337, add_435);  view_337 = add_435 = None
        sum_149 = torch.ops.aten.sum.dim_IntList(mul_748, [0, 1, 2], True)
        view_338 = torch.ops.aten.view.default(sum_149, [1152]);  sum_149 = None
        view_339 = torch.ops.aten.view.default(mul_748, [12544, 1152]);  mul_748 = None
        permute_425 = torch.ops.aten.permute.default(view_339, [1, 0])
        mm_171 = torch.ops.aten.mm.default(permute_425, view_66);  permute_425 = view_66 = None
        permute_426 = torch.ops.aten.permute.default(mm_171, [1, 0]);  mm_171 = None
        mm_172 = torch.ops.aten.mm.default(view_339, permute_427);  view_339 = permute_427 = None
        view_340 = torch.ops.aten.view.default(mm_172, [64, 14, 14, 384]);  mm_172 = None
        permute_428 = torch.ops.aten.permute.default(permute_426, [1, 0]);  permute_426 = None
        mul_750 = torch.ops.aten.mul.Tensor(view_340, primals_128);  primals_128 = None
        mul_751 = torch.ops.aten.mul.Tensor(mul_750, 384)
        sum_150 = torch.ops.aten.sum.dim_IntList(mul_750, [3], True)
        mul_752 = torch.ops.aten.mul.Tensor(mul_750, mul_186);  mul_750 = None
        sum_151 = torch.ops.aten.sum.dim_IntList(mul_752, [3], True);  mul_752 = None
        mul_753 = torch.ops.aten.mul.Tensor(mul_186, sum_151);  sum_151 = None
        sub_169 = torch.ops.aten.sub.Tensor(mul_751, sum_150);  mul_751 = sum_150 = None
        sub_170 = torch.ops.aten.sub.Tensor(sub_169, mul_753);  sub_169 = mul_753 = None
        div_41 = torch.ops.aten.div.Tensor(reciprocal_31, 384);  reciprocal_31 = None
        mul_754 = torch.ops.aten.mul.Tensor(div_41, sub_170);  div_41 = sub_170 = None
        mul_755 = torch.ops.aten.mul.Tensor(view_340, mul_186);  mul_186 = None
        sum_152 = torch.ops.aten.sum.dim_IntList(mul_755, [0, 1, 2]);  mul_755 = None
        sum_153 = torch.ops.aten.sum.dim_IntList(view_340, [0, 1, 2]);  view_340 = None
        add_436 = torch.ops.aten.add.Tensor(add_428, mul_754);  add_428 = mul_754 = None
        sum_154 = torch.ops.aten.sum.dim_IntList(add_436, [0, 1, 2], True)
        view_341 = torch.ops.aten.view.default(sum_154, [384]);  sum_154 = None
        view_342 = torch.ops.aten.view.default(add_436, [12544, 384])
        permute_429 = torch.ops.aten.permute.default(view_342, [1, 0])
        mm_173 = torch.ops.aten.mm.default(permute_429, view_65);  permute_429 = view_65 = None
        permute_430 = torch.ops.aten.permute.default(mm_173, [1, 0]);  mm_173 = None
        mm_174 = torch.ops.aten.mm.default(view_342, permute_431);  view_342 = permute_431 = None
        view_343 = torch.ops.aten.view.default(mm_174, [64, 14, 14, 384]);  mm_174 = None
        permute_432 = torch.ops.aten.permute.default(permute_430, [1, 0]);  permute_430 = None
        view_344 = torch.ops.aten.view.default(view_343, [64, 196, 12, 32]);  view_343 = None
        permute_433 = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
        clone_153 = torch.ops.aten.clone.default(permute_433, memory_format = torch.contiguous_format);  permute_433 = None
        _unsafe_view_213 = torch.ops.aten._unsafe_view.default(clone_153, [768, 196, 32]);  clone_153 = None
        permute_434 = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
        bmm_76 = torch.ops.aten.bmm.default(permute_434, _unsafe_view_213);  permute_434 = None
        bmm_77 = torch.ops.aten.bmm.default(_unsafe_view_213, permute_435);  _unsafe_view_213 = permute_435 = None
        view_345 = torch.ops.aten.view.default(bmm_76, [64, 12, 196, 32]);  bmm_76 = None
        view_346 = torch.ops.aten.view.default(bmm_77, [64, 12, 196, 196]);  bmm_77 = None
        alias_121 = torch.ops.aten.alias.default(alias_70);  alias_70 = None
        alias_122 = torch.ops.aten.alias.default(alias_121);  alias_121 = None
        mul_756 = torch.ops.aten.mul.Tensor(view_346, alias_122);  view_346 = None
        sum_155 = torch.ops.aten.sum.dim_IntList(mul_756, [-1], True)
        mul_757 = torch.ops.aten.mul.Tensor(alias_122, sum_155);  alias_122 = sum_155 = None
        sub_171 = torch.ops.aten.sub.Tensor(mul_756, mul_757);  mul_756 = mul_757 = None
        mul_758 = torch.ops.aten.mul.Tensor(sub_171, 0.1767766952966369);  sub_171 = None
        view_347 = torch.ops.aten.view.default(mul_758, [768, 196, 196]);  mul_758 = None
        bmm_78 = torch.ops.aten.bmm.default(permute_436, view_347);  permute_436 = None
        bmm_79 = torch.ops.aten.bmm.default(view_347, permute_437);  view_347 = permute_437 = None
        view_348 = torch.ops.aten.view.default(bmm_78, [64, 12, 32, 196]);  bmm_78 = None
        view_349 = torch.ops.aten.view.default(bmm_79, [64, 12, 196, 32]);  bmm_79 = None
        permute_438 = torch.ops.aten.permute.default(view_348, [0, 1, 3, 2]);  view_348 = None
        cat_13 = torch.ops.aten.cat.default([view_349, permute_438, view_345]);  view_349 = permute_438 = view_345 = None
        view_350 = torch.ops.aten.view.default(cat_13, [3, 64, 12, 196, 32]);  cat_13 = None
        permute_439 = torch.ops.aten.permute.default(view_350, [1, 3, 0, 2, 4]);  view_350 = None
        clone_154 = torch.ops.aten.clone.default(permute_439, memory_format = torch.contiguous_format);  permute_439 = None
        _unsafe_view_214 = torch.ops.aten._unsafe_view.default(clone_154, [64, 14, 14, 1152]);  clone_154 = None
        view_351 = torch.ops.aten.view.default(_unsafe_view_214, [12544, 1152]);  _unsafe_view_214 = None
        permute_440 = torch.ops.aten.permute.default(view_351, [1, 0])
        mm_175 = torch.ops.aten.mm.default(permute_440, view_62);  permute_440 = view_62 = None
        permute_441 = torch.ops.aten.permute.default(mm_175, [1, 0]);  mm_175 = None
        mm_176 = torch.ops.aten.mm.default(view_351, permute_442);  view_351 = permute_442 = None
        view_352 = torch.ops.aten.view.default(mm_176, [64, 14, 14, 384]);  mm_176 = None
        permute_443 = torch.ops.aten.permute.default(permute_441, [1, 0]);  permute_441 = None
        mul_760 = torch.ops.aten.mul.Tensor(view_352, primals_123);  primals_123 = None
        mul_761 = torch.ops.aten.mul.Tensor(mul_760, 384)
        sum_156 = torch.ops.aten.sum.dim_IntList(mul_760, [3], True)
        mul_762 = torch.ops.aten.mul.Tensor(mul_760, mul_183);  mul_760 = None
        sum_157 = torch.ops.aten.sum.dim_IntList(mul_762, [3], True);  mul_762 = None
        mul_763 = torch.ops.aten.mul.Tensor(mul_183, sum_157);  sum_157 = None
        sub_173 = torch.ops.aten.sub.Tensor(mul_761, sum_156);  mul_761 = sum_156 = None
        sub_174 = torch.ops.aten.sub.Tensor(sub_173, mul_763);  sub_173 = mul_763 = None
        div_42 = torch.ops.aten.div.Tensor(reciprocal_30, 384);  reciprocal_30 = None
        mul_764 = torch.ops.aten.mul.Tensor(div_42, sub_174);  div_42 = sub_174 = None
        mul_765 = torch.ops.aten.mul.Tensor(view_352, mul_183);  mul_183 = None
        sum_158 = torch.ops.aten.sum.dim_IntList(mul_765, [0, 1, 2]);  mul_765 = None
        sum_159 = torch.ops.aten.sum.dim_IntList(view_352, [0, 1, 2]);  view_352 = None
        add_437 = torch.ops.aten.add.Tensor(add_436, mul_764);  add_436 = mul_764 = None
        sum_160 = torch.ops.aten.sum.dim_IntList(add_437, [0, 1, 2], True)
        view_353 = torch.ops.aten.view.default(sum_160, [384]);  sum_160 = None
        view_354 = torch.ops.aten.view.default(add_437, [12544, 384])
        permute_444 = torch.ops.aten.permute.default(view_354, [1, 0])
        mm_177 = torch.ops.aten.mm.default(permute_444, view_61);  permute_444 = view_61 = None
        permute_445 = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
        mm_178 = torch.ops.aten.mm.default(view_354, permute_446);  view_354 = permute_446 = None
        view_355 = torch.ops.aten.view.default(mm_178, [64, 14, 14, 1152]);  mm_178 = None
        permute_447 = torch.ops.aten.permute.default(permute_445, [1, 0]);  permute_445 = None
        mul_782 = torch.ops.aten.mul.Tensor(view_355, add_444);  view_355 = add_444 = None
        sum_161 = torch.ops.aten.sum.dim_IntList(mul_782, [0, 1, 2], True)
        view_356 = torch.ops.aten.view.default(sum_161, [1152]);  sum_161 = None
        view_357 = torch.ops.aten.view.default(mul_782, [12544, 1152]);  mul_782 = None
        permute_448 = torch.ops.aten.permute.default(view_357, [1, 0])
        mm_179 = torch.ops.aten.mm.default(permute_448, view_60);  permute_448 = view_60 = None
        permute_449 = torch.ops.aten.permute.default(mm_179, [1, 0]);  mm_179 = None
        mm_180 = torch.ops.aten.mm.default(view_357, permute_450);  view_357 = permute_450 = None
        view_358 = torch.ops.aten.view.default(mm_180, [64, 14, 14, 384]);  mm_180 = None
        permute_451 = torch.ops.aten.permute.default(permute_449, [1, 0]);  permute_449 = None
        mul_784 = torch.ops.aten.mul.Tensor(view_358, primals_117);  primals_117 = None
        mul_785 = torch.ops.aten.mul.Tensor(mul_784, 384)
        sum_162 = torch.ops.aten.sum.dim_IntList(mul_784, [3], True)
        mul_786 = torch.ops.aten.mul.Tensor(mul_784, mul_168);  mul_784 = None
        sum_163 = torch.ops.aten.sum.dim_IntList(mul_786, [3], True);  mul_786 = None
        mul_787 = torch.ops.aten.mul.Tensor(mul_168, sum_163);  sum_163 = None
        sub_177 = torch.ops.aten.sub.Tensor(mul_785, sum_162);  mul_785 = sum_162 = None
        sub_178 = torch.ops.aten.sub.Tensor(sub_177, mul_787);  sub_177 = mul_787 = None
        div_43 = torch.ops.aten.div.Tensor(reciprocal_28, 384);  reciprocal_28 = None
        mul_788 = torch.ops.aten.mul.Tensor(div_43, sub_178);  div_43 = sub_178 = None
        mul_789 = torch.ops.aten.mul.Tensor(view_358, mul_168);  mul_168 = None
        sum_164 = torch.ops.aten.sum.dim_IntList(mul_789, [0, 1, 2]);  mul_789 = None
        sum_165 = torch.ops.aten.sum.dim_IntList(view_358, [0, 1, 2]);  view_358 = None
        add_445 = torch.ops.aten.add.Tensor(add_437, mul_788);  add_437 = mul_788 = None
        sum_166 = torch.ops.aten.sum.dim_IntList(add_445, [0, 1, 2], True)
        view_359 = torch.ops.aten.view.default(sum_166, [384]);  sum_166 = None
        view_360 = torch.ops.aten.view.default(add_445, [12544, 384])
        permute_452 = torch.ops.aten.permute.default(view_360, [1, 0])
        mm_181 = torch.ops.aten.mm.default(permute_452, view_59);  permute_452 = view_59 = None
        permute_453 = torch.ops.aten.permute.default(mm_181, [1, 0]);  mm_181 = None
        mm_182 = torch.ops.aten.mm.default(view_360, permute_454);  view_360 = permute_454 = None
        view_361 = torch.ops.aten.view.default(mm_182, [64, 14, 14, 384]);  mm_182 = None
        permute_455 = torch.ops.aten.permute.default(permute_453, [1, 0]);  permute_453 = None
        view_362 = torch.ops.aten.view.default(view_361, [64, 196, 12, 32]);  view_361 = None
        permute_456 = torch.ops.aten.permute.default(view_362, [0, 2, 1, 3]);  view_362 = None
        clone_157 = torch.ops.aten.clone.default(permute_456, memory_format = torch.contiguous_format);  permute_456 = None
        _unsafe_view_215 = torch.ops.aten._unsafe_view.default(clone_157, [768, 196, 32]);  clone_157 = None
        permute_457 = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
        bmm_80 = torch.ops.aten.bmm.default(permute_457, _unsafe_view_215);  permute_457 = None
        bmm_81 = torch.ops.aten.bmm.default(_unsafe_view_215, permute_458);  _unsafe_view_215 = permute_458 = None
        view_363 = torch.ops.aten.view.default(bmm_80, [64, 12, 196, 32]);  bmm_80 = None
        view_364 = torch.ops.aten.view.default(bmm_81, [64, 12, 196, 196]);  bmm_81 = None
        alias_123 = torch.ops.aten.alias.default(alias_67);  alias_67 = None
        alias_124 = torch.ops.aten.alias.default(alias_123);  alias_123 = None
        mul_790 = torch.ops.aten.mul.Tensor(view_364, alias_124);  view_364 = None
        sum_167 = torch.ops.aten.sum.dim_IntList(mul_790, [-1], True)
        mul_791 = torch.ops.aten.mul.Tensor(alias_124, sum_167);  alias_124 = sum_167 = None
        sub_179 = torch.ops.aten.sub.Tensor(mul_790, mul_791);  mul_790 = mul_791 = None
        mul_792 = torch.ops.aten.mul.Tensor(sub_179, 0.1767766952966369);  sub_179 = None
        view_365 = torch.ops.aten.view.default(mul_792, [768, 196, 196]);  mul_792 = None
        bmm_82 = torch.ops.aten.bmm.default(permute_459, view_365);  permute_459 = None
        bmm_83 = torch.ops.aten.bmm.default(view_365, permute_460);  view_365 = permute_460 = None
        view_366 = torch.ops.aten.view.default(bmm_82, [64, 12, 32, 196]);  bmm_82 = None
        view_367 = torch.ops.aten.view.default(bmm_83, [64, 12, 196, 32]);  bmm_83 = None
        permute_461 = torch.ops.aten.permute.default(view_366, [0, 1, 3, 2]);  view_366 = None
        cat_14 = torch.ops.aten.cat.default([view_367, permute_461, view_363]);  view_367 = permute_461 = view_363 = None
        view_368 = torch.ops.aten.view.default(cat_14, [3, 64, 12, 196, 32]);  cat_14 = None
        permute_462 = torch.ops.aten.permute.default(view_368, [1, 3, 0, 2, 4]);  view_368 = None
        clone_158 = torch.ops.aten.clone.default(permute_462, memory_format = torch.contiguous_format);  permute_462 = None
        _unsafe_view_216 = torch.ops.aten._unsafe_view.default(clone_158, [64, 14, 14, 1152]);  clone_158 = None
        view_369 = torch.ops.aten.view.default(_unsafe_view_216, [12544, 1152]);  _unsafe_view_216 = None
        permute_463 = torch.ops.aten.permute.default(view_369, [1, 0])
        mm_183 = torch.ops.aten.mm.default(permute_463, view_56);  permute_463 = view_56 = None
        permute_464 = torch.ops.aten.permute.default(mm_183, [1, 0]);  mm_183 = None
        mm_184 = torch.ops.aten.mm.default(view_369, permute_465);  view_369 = permute_465 = None
        view_370 = torch.ops.aten.view.default(mm_184, [64, 14, 14, 384]);  mm_184 = None
        permute_466 = torch.ops.aten.permute.default(permute_464, [1, 0]);  permute_464 = None
        mul_794 = torch.ops.aten.mul.Tensor(view_370, primals_112);  primals_112 = None
        mul_795 = torch.ops.aten.mul.Tensor(mul_794, 384)
        sum_168 = torch.ops.aten.sum.dim_IntList(mul_794, [3], True)
        mul_796 = torch.ops.aten.mul.Tensor(mul_794, mul_165);  mul_794 = None
        sum_169 = torch.ops.aten.sum.dim_IntList(mul_796, [3], True);  mul_796 = None
        mul_797 = torch.ops.aten.mul.Tensor(mul_165, sum_169);  sum_169 = None
        sub_181 = torch.ops.aten.sub.Tensor(mul_795, sum_168);  mul_795 = sum_168 = None
        sub_182 = torch.ops.aten.sub.Tensor(sub_181, mul_797);  sub_181 = mul_797 = None
        div_44 = torch.ops.aten.div.Tensor(reciprocal_27, 384);  reciprocal_27 = None
        mul_798 = torch.ops.aten.mul.Tensor(div_44, sub_182);  div_44 = sub_182 = None
        mul_799 = torch.ops.aten.mul.Tensor(view_370, mul_165);  mul_165 = None
        sum_170 = torch.ops.aten.sum.dim_IntList(mul_799, [0, 1, 2]);  mul_799 = None
        sum_171 = torch.ops.aten.sum.dim_IntList(view_370, [0, 1, 2]);  view_370 = None
        add_446 = torch.ops.aten.add.Tensor(add_445, mul_798);  add_445 = mul_798 = None
        sum_172 = torch.ops.aten.sum.dim_IntList(add_446, [0, 1, 2], True)
        view_371 = torch.ops.aten.view.default(sum_172, [384]);  sum_172 = None
        view_372 = torch.ops.aten.view.default(add_446, [12544, 384])
        permute_467 = torch.ops.aten.permute.default(view_372, [1, 0])
        mm_185 = torch.ops.aten.mm.default(permute_467, view_55);  permute_467 = view_55 = None
        permute_468 = torch.ops.aten.permute.default(mm_185, [1, 0]);  mm_185 = None
        mm_186 = torch.ops.aten.mm.default(view_372, permute_469);  view_372 = permute_469 = None
        view_373 = torch.ops.aten.view.default(mm_186, [64, 14, 14, 1152]);  mm_186 = None
        permute_470 = torch.ops.aten.permute.default(permute_468, [1, 0]);  permute_468 = None
        mul_816 = torch.ops.aten.mul.Tensor(view_373, add_453);  view_373 = add_453 = None
        sum_173 = torch.ops.aten.sum.dim_IntList(mul_816, [0, 1, 2], True)
        view_374 = torch.ops.aten.view.default(sum_173, [1152]);  sum_173 = None
        view_375 = torch.ops.aten.view.default(mul_816, [12544, 1152]);  mul_816 = None
        permute_471 = torch.ops.aten.permute.default(view_375, [1, 0])
        mm_187 = torch.ops.aten.mm.default(permute_471, view_54);  permute_471 = view_54 = None
        permute_472 = torch.ops.aten.permute.default(mm_187, [1, 0]);  mm_187 = None
        mm_188 = torch.ops.aten.mm.default(view_375, permute_473);  view_375 = permute_473 = None
        view_376 = torch.ops.aten.view.default(mm_188, [64, 14, 14, 384]);  mm_188 = None
        permute_474 = torch.ops.aten.permute.default(permute_472, [1, 0]);  permute_472 = None
        mul_818 = torch.ops.aten.mul.Tensor(view_376, primals_106);  primals_106 = None
        mul_819 = torch.ops.aten.mul.Tensor(mul_818, 384)
        sum_174 = torch.ops.aten.sum.dim_IntList(mul_818, [3], True)
        mul_820 = torch.ops.aten.mul.Tensor(mul_818, mul_150);  mul_818 = None
        sum_175 = torch.ops.aten.sum.dim_IntList(mul_820, [3], True);  mul_820 = None
        mul_821 = torch.ops.aten.mul.Tensor(mul_150, sum_175);  sum_175 = None
        sub_185 = torch.ops.aten.sub.Tensor(mul_819, sum_174);  mul_819 = sum_174 = None
        sub_186 = torch.ops.aten.sub.Tensor(sub_185, mul_821);  sub_185 = mul_821 = None
        div_45 = torch.ops.aten.div.Tensor(reciprocal_25, 384);  reciprocal_25 = None
        mul_822 = torch.ops.aten.mul.Tensor(div_45, sub_186);  div_45 = sub_186 = None
        mul_823 = torch.ops.aten.mul.Tensor(view_376, mul_150);  mul_150 = None
        sum_176 = torch.ops.aten.sum.dim_IntList(mul_823, [0, 1, 2]);  mul_823 = None
        sum_177 = torch.ops.aten.sum.dim_IntList(view_376, [0, 1, 2]);  view_376 = None
        add_454 = torch.ops.aten.add.Tensor(add_446, mul_822);  add_446 = mul_822 = None
        sum_178 = torch.ops.aten.sum.dim_IntList(add_454, [0, 1, 2], True)
        view_377 = torch.ops.aten.view.default(sum_178, [384]);  sum_178 = None
        view_378 = torch.ops.aten.view.default(add_454, [12544, 384])
        permute_475 = torch.ops.aten.permute.default(view_378, [1, 0])
        mm_189 = torch.ops.aten.mm.default(permute_475, view_53);  permute_475 = view_53 = None
        permute_476 = torch.ops.aten.permute.default(mm_189, [1, 0]);  mm_189 = None
        mm_190 = torch.ops.aten.mm.default(view_378, permute_477);  view_378 = permute_477 = None
        view_379 = torch.ops.aten.view.default(mm_190, [64, 14, 14, 384]);  mm_190 = None
        permute_478 = torch.ops.aten.permute.default(permute_476, [1, 0]);  permute_476 = None
        view_380 = torch.ops.aten.view.default(view_379, [64, 196, 12, 32]);  view_379 = None
        permute_479 = torch.ops.aten.permute.default(view_380, [0, 2, 1, 3]);  view_380 = None
        clone_161 = torch.ops.aten.clone.default(permute_479, memory_format = torch.contiguous_format);  permute_479 = None
        _unsafe_view_217 = torch.ops.aten._unsafe_view.default(clone_161, [768, 196, 32]);  clone_161 = None
        permute_480 = torch.ops.aten.permute.default(view_52, [0, 2, 1]);  view_52 = None
        bmm_84 = torch.ops.aten.bmm.default(permute_480, _unsafe_view_217);  permute_480 = None
        bmm_85 = torch.ops.aten.bmm.default(_unsafe_view_217, permute_481);  _unsafe_view_217 = permute_481 = None
        view_381 = torch.ops.aten.view.default(bmm_84, [64, 12, 196, 32]);  bmm_84 = None
        view_382 = torch.ops.aten.view.default(bmm_85, [64, 12, 196, 196]);  bmm_85 = None
        alias_125 = torch.ops.aten.alias.default(alias_64);  alias_64 = None
        alias_126 = torch.ops.aten.alias.default(alias_125);  alias_125 = None
        mul_824 = torch.ops.aten.mul.Tensor(view_382, alias_126);  view_382 = None
        sum_179 = torch.ops.aten.sum.dim_IntList(mul_824, [-1], True)
        mul_825 = torch.ops.aten.mul.Tensor(alias_126, sum_179);  alias_126 = sum_179 = None
        sub_187 = torch.ops.aten.sub.Tensor(mul_824, mul_825);  mul_824 = mul_825 = None
        mul_826 = torch.ops.aten.mul.Tensor(sub_187, 0.1767766952966369);  sub_187 = None
        view_383 = torch.ops.aten.view.default(mul_826, [768, 196, 196]);  mul_826 = None
        bmm_86 = torch.ops.aten.bmm.default(permute_482, view_383);  permute_482 = None
        bmm_87 = torch.ops.aten.bmm.default(view_383, permute_483);  view_383 = permute_483 = None
        view_384 = torch.ops.aten.view.default(bmm_86, [64, 12, 32, 196]);  bmm_86 = None
        view_385 = torch.ops.aten.view.default(bmm_87, [64, 12, 196, 32]);  bmm_87 = None
        permute_484 = torch.ops.aten.permute.default(view_384, [0, 1, 3, 2]);  view_384 = None
        cat_15 = torch.ops.aten.cat.default([view_385, permute_484, view_381]);  view_385 = permute_484 = view_381 = None
        view_386 = torch.ops.aten.view.default(cat_15, [3, 64, 12, 196, 32]);  cat_15 = None
        permute_485 = torch.ops.aten.permute.default(view_386, [1, 3, 0, 2, 4]);  view_386 = None
        clone_162 = torch.ops.aten.clone.default(permute_485, memory_format = torch.contiguous_format);  permute_485 = None
        _unsafe_view_218 = torch.ops.aten._unsafe_view.default(clone_162, [64, 14, 14, 1152]);  clone_162 = None
        view_387 = torch.ops.aten.view.default(_unsafe_view_218, [12544, 1152]);  _unsafe_view_218 = None
        permute_486 = torch.ops.aten.permute.default(view_387, [1, 0])
        mm_191 = torch.ops.aten.mm.default(permute_486, view_50);  permute_486 = view_50 = None
        permute_487 = torch.ops.aten.permute.default(mm_191, [1, 0]);  mm_191 = None
        mm_192 = torch.ops.aten.mm.default(view_387, permute_488);  view_387 = permute_488 = None
        view_388 = torch.ops.aten.view.default(mm_192, [64, 14, 14, 384]);  mm_192 = None
        permute_489 = torch.ops.aten.permute.default(permute_487, [1, 0]);  permute_487 = None
        mul_828 = torch.ops.aten.mul.Tensor(view_388, primals_101);  primals_101 = None
        mul_829 = torch.ops.aten.mul.Tensor(mul_828, 384)
        sum_180 = torch.ops.aten.sum.dim_IntList(mul_828, [3], True)
        mul_830 = torch.ops.aten.mul.Tensor(mul_828, mul_147);  mul_828 = None
        sum_181 = torch.ops.aten.sum.dim_IntList(mul_830, [3], True);  mul_830 = None
        mul_831 = torch.ops.aten.mul.Tensor(mul_147, sum_181);  sum_181 = None
        sub_189 = torch.ops.aten.sub.Tensor(mul_829, sum_180);  mul_829 = sum_180 = None
        sub_190 = torch.ops.aten.sub.Tensor(sub_189, mul_831);  sub_189 = mul_831 = None
        div_46 = torch.ops.aten.div.Tensor(reciprocal_24, 384);  reciprocal_24 = None
        mul_832 = torch.ops.aten.mul.Tensor(div_46, sub_190);  div_46 = sub_190 = None
        mul_833 = torch.ops.aten.mul.Tensor(view_388, mul_147);  mul_147 = None
        sum_182 = torch.ops.aten.sum.dim_IntList(mul_833, [0, 1, 2]);  mul_833 = None
        sum_183 = torch.ops.aten.sum.dim_IntList(view_388, [0, 1, 2]);  view_388 = None
        add_455 = torch.ops.aten.add.Tensor(add_454, mul_832);  add_454 = mul_832 = None
        sum_184 = torch.ops.aten.sum.dim_IntList(add_455, [0, 1, 2], True)
        view_389 = torch.ops.aten.view.default(sum_184, [384]);  sum_184 = None
        view_390 = torch.ops.aten.view.default(add_455, [12544, 384])
        permute_490 = torch.ops.aten.permute.default(view_390, [1, 0])
        mm_193 = torch.ops.aten.mm.default(permute_490, view_49);  permute_490 = view_49 = None
        permute_491 = torch.ops.aten.permute.default(mm_193, [1, 0]);  mm_193 = None
        mm_194 = torch.ops.aten.mm.default(view_390, permute_492);  view_390 = permute_492 = None
        view_391 = torch.ops.aten.view.default(mm_194, [64, 14, 14, 1152]);  mm_194 = None
        permute_493 = torch.ops.aten.permute.default(permute_491, [1, 0]);  permute_491 = None
        mul_850 = torch.ops.aten.mul.Tensor(view_391, add_462);  view_391 = add_462 = None
        sum_185 = torch.ops.aten.sum.dim_IntList(mul_850, [0, 1, 2], True)
        view_392 = torch.ops.aten.view.default(sum_185, [1152]);  sum_185 = None
        view_393 = torch.ops.aten.view.default(mul_850, [12544, 1152]);  mul_850 = None
        permute_494 = torch.ops.aten.permute.default(view_393, [1, 0])
        mm_195 = torch.ops.aten.mm.default(permute_494, view_48);  permute_494 = view_48 = None
        permute_495 = torch.ops.aten.permute.default(mm_195, [1, 0]);  mm_195 = None
        mm_196 = torch.ops.aten.mm.default(view_393, permute_496);  view_393 = permute_496 = None
        view_394 = torch.ops.aten.view.default(mm_196, [64, 14, 14, 384]);  mm_196 = None
        permute_497 = torch.ops.aten.permute.default(permute_495, [1, 0]);  permute_495 = None
        mul_852 = torch.ops.aten.mul.Tensor(view_394, primals_95);  primals_95 = None
        mul_853 = torch.ops.aten.mul.Tensor(mul_852, 384)
        sum_186 = torch.ops.aten.sum.dim_IntList(mul_852, [3], True)
        mul_854 = torch.ops.aten.mul.Tensor(mul_852, mul_132);  mul_852 = None
        sum_187 = torch.ops.aten.sum.dim_IntList(mul_854, [3], True);  mul_854 = None
        mul_855 = torch.ops.aten.mul.Tensor(mul_132, sum_187);  sum_187 = None
        sub_193 = torch.ops.aten.sub.Tensor(mul_853, sum_186);  mul_853 = sum_186 = None
        sub_194 = torch.ops.aten.sub.Tensor(sub_193, mul_855);  sub_193 = mul_855 = None
        div_47 = torch.ops.aten.div.Tensor(reciprocal_22, 384);  reciprocal_22 = None
        mul_856 = torch.ops.aten.mul.Tensor(div_47, sub_194);  div_47 = sub_194 = None
        mul_857 = torch.ops.aten.mul.Tensor(view_394, mul_132);  mul_132 = None
        sum_188 = torch.ops.aten.sum.dim_IntList(mul_857, [0, 1, 2]);  mul_857 = None
        sum_189 = torch.ops.aten.sum.dim_IntList(view_394, [0, 1, 2]);  view_394 = None
        add_463 = torch.ops.aten.add.Tensor(add_455, mul_856);  add_455 = mul_856 = None
        sum_190 = torch.ops.aten.sum.dim_IntList(add_463, [0, 1, 2], True)
        view_395 = torch.ops.aten.view.default(sum_190, [384]);  sum_190 = None
        view_396 = torch.ops.aten.view.default(add_463, [12544, 384])
        permute_498 = torch.ops.aten.permute.default(view_396, [1, 0])
        mm_197 = torch.ops.aten.mm.default(permute_498, view_47);  permute_498 = view_47 = None
        permute_499 = torch.ops.aten.permute.default(mm_197, [1, 0]);  mm_197 = None
        mm_198 = torch.ops.aten.mm.default(view_396, permute_500);  view_396 = permute_500 = None
        view_397 = torch.ops.aten.view.default(mm_198, [64, 14, 14, 384]);  mm_198 = None
        permute_501 = torch.ops.aten.permute.default(permute_499, [1, 0]);  permute_499 = None
        view_398 = torch.ops.aten.view.default(view_397, [64, 196, 12, 32]);  view_397 = None
        permute_502 = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None
        clone_165 = torch.ops.aten.clone.default(permute_502, memory_format = torch.contiguous_format);  permute_502 = None
        _unsafe_view_219 = torch.ops.aten._unsafe_view.default(clone_165, [768, 196, 32]);  clone_165 = None
        permute_503 = torch.ops.aten.permute.default(view_46, [0, 2, 1]);  view_46 = None
        bmm_88 = torch.ops.aten.bmm.default(permute_503, _unsafe_view_219);  permute_503 = None
        bmm_89 = torch.ops.aten.bmm.default(_unsafe_view_219, permute_504);  _unsafe_view_219 = permute_504 = None
        view_399 = torch.ops.aten.view.default(bmm_88, [64, 12, 196, 32]);  bmm_88 = None
        view_400 = torch.ops.aten.view.default(bmm_89, [64, 12, 196, 196]);  bmm_89 = None
        alias_127 = torch.ops.aten.alias.default(alias_61);  alias_61 = None
        alias_128 = torch.ops.aten.alias.default(alias_127);  alias_127 = None
        mul_858 = torch.ops.aten.mul.Tensor(view_400, alias_128);  view_400 = None
        sum_191 = torch.ops.aten.sum.dim_IntList(mul_858, [-1], True)
        mul_859 = torch.ops.aten.mul.Tensor(alias_128, sum_191);  alias_128 = sum_191 = None
        sub_195 = torch.ops.aten.sub.Tensor(mul_858, mul_859);  mul_858 = mul_859 = None
        mul_860 = torch.ops.aten.mul.Tensor(sub_195, 0.1767766952966369);  sub_195 = None
        view_401 = torch.ops.aten.view.default(mul_860, [768, 196, 196]);  mul_860 = None
        bmm_90 = torch.ops.aten.bmm.default(permute_505, view_401);  permute_505 = None
        bmm_91 = torch.ops.aten.bmm.default(view_401, permute_506);  view_401 = permute_506 = None
        view_402 = torch.ops.aten.view.default(bmm_90, [64, 12, 32, 196]);  bmm_90 = None
        view_403 = torch.ops.aten.view.default(bmm_91, [64, 12, 196, 32]);  bmm_91 = None
        permute_507 = torch.ops.aten.permute.default(view_402, [0, 1, 3, 2]);  view_402 = None
        cat_16 = torch.ops.aten.cat.default([view_403, permute_507, view_399]);  view_403 = permute_507 = view_399 = None
        view_404 = torch.ops.aten.view.default(cat_16, [3, 64, 12, 196, 32]);  cat_16 = None
        permute_508 = torch.ops.aten.permute.default(view_404, [1, 3, 0, 2, 4]);  view_404 = None
        clone_166 = torch.ops.aten.clone.default(permute_508, memory_format = torch.contiguous_format);  permute_508 = None
        _unsafe_view_220 = torch.ops.aten._unsafe_view.default(clone_166, [64, 14, 14, 1152]);  clone_166 = None
        view_405 = torch.ops.aten.view.default(_unsafe_view_220, [12544, 1152]);  _unsafe_view_220 = None
        permute_509 = torch.ops.aten.permute.default(view_405, [1, 0])
        mm_199 = torch.ops.aten.mm.default(permute_509, view_44);  permute_509 = view_44 = None
        permute_510 = torch.ops.aten.permute.default(mm_199, [1, 0]);  mm_199 = None
        mm_200 = torch.ops.aten.mm.default(view_405, permute_511);  view_405 = permute_511 = None
        view_406 = torch.ops.aten.view.default(mm_200, [64, 14, 14, 384]);  mm_200 = None
        permute_512 = torch.ops.aten.permute.default(permute_510, [1, 0]);  permute_510 = None
        mul_862 = torch.ops.aten.mul.Tensor(view_406, primals_90);  primals_90 = None
        mul_863 = torch.ops.aten.mul.Tensor(mul_862, 384)
        sum_192 = torch.ops.aten.sum.dim_IntList(mul_862, [3], True)
        mul_864 = torch.ops.aten.mul.Tensor(mul_862, mul_129);  mul_862 = None
        sum_193 = torch.ops.aten.sum.dim_IntList(mul_864, [3], True);  mul_864 = None
        mul_865 = torch.ops.aten.mul.Tensor(mul_129, sum_193);  sum_193 = None
        sub_197 = torch.ops.aten.sub.Tensor(mul_863, sum_192);  mul_863 = sum_192 = None
        sub_198 = torch.ops.aten.sub.Tensor(sub_197, mul_865);  sub_197 = mul_865 = None
        div_48 = torch.ops.aten.div.Tensor(reciprocal_21, 384);  reciprocal_21 = None
        mul_866 = torch.ops.aten.mul.Tensor(div_48, sub_198);  div_48 = sub_198 = None
        mul_867 = torch.ops.aten.mul.Tensor(view_406, mul_129);  mul_129 = None
        sum_194 = torch.ops.aten.sum.dim_IntList(mul_867, [0, 1, 2]);  mul_867 = None
        sum_195 = torch.ops.aten.sum.dim_IntList(view_406, [0, 1, 2]);  view_406 = None
        add_464 = torch.ops.aten.add.Tensor(add_463, mul_866);  add_463 = mul_866 = None
        sum_196 = torch.ops.aten.sum.dim_IntList(add_464, [0, 1, 2], True)
        view_407 = torch.ops.aten.view.default(sum_196, [384]);  sum_196 = None
        view_408 = torch.ops.aten.view.default(add_464, [12544, 384])
        permute_513 = torch.ops.aten.permute.default(view_408, [1, 0])
        mm_201 = torch.ops.aten.mm.default(permute_513, view_43);  permute_513 = view_43 = None
        permute_514 = torch.ops.aten.permute.default(mm_201, [1, 0]);  mm_201 = None
        mm_202 = torch.ops.aten.mm.default(view_408, permute_515);  view_408 = permute_515 = None
        view_409 = torch.ops.aten.view.default(mm_202, [64, 14, 14, 1152]);  mm_202 = None
        permute_516 = torch.ops.aten.permute.default(permute_514, [1, 0]);  permute_514 = None
        mul_884 = torch.ops.aten.mul.Tensor(view_409, add_471);  view_409 = add_471 = None
        sum_197 = torch.ops.aten.sum.dim_IntList(mul_884, [0, 1, 2], True)
        view_410 = torch.ops.aten.view.default(sum_197, [1152]);  sum_197 = None
        view_411 = torch.ops.aten.view.default(mul_884, [12544, 1152]);  mul_884 = None
        permute_517 = torch.ops.aten.permute.default(view_411, [1, 0])
        mm_203 = torch.ops.aten.mm.default(permute_517, view_42);  permute_517 = view_42 = None
        permute_518 = torch.ops.aten.permute.default(mm_203, [1, 0]);  mm_203 = None
        mm_204 = torch.ops.aten.mm.default(view_411, permute_519);  view_411 = permute_519 = None
        view_412 = torch.ops.aten.view.default(mm_204, [64, 14, 14, 384]);  mm_204 = None
        permute_520 = torch.ops.aten.permute.default(permute_518, [1, 0]);  permute_518 = None
        mul_886 = torch.ops.aten.mul.Tensor(view_412, primals_84);  primals_84 = None
        mul_887 = torch.ops.aten.mul.Tensor(mul_886, 384)
        sum_198 = torch.ops.aten.sum.dim_IntList(mul_886, [3], True)
        mul_888 = torch.ops.aten.mul.Tensor(mul_886, mul_114);  mul_886 = None
        sum_199 = torch.ops.aten.sum.dim_IntList(mul_888, [3], True);  mul_888 = None
        mul_889 = torch.ops.aten.mul.Tensor(mul_114, sum_199);  sum_199 = None
        sub_201 = torch.ops.aten.sub.Tensor(mul_887, sum_198);  mul_887 = sum_198 = None
        sub_202 = torch.ops.aten.sub.Tensor(sub_201, mul_889);  sub_201 = mul_889 = None
        div_49 = torch.ops.aten.div.Tensor(reciprocal_19, 384);  reciprocal_19 = None
        mul_890 = torch.ops.aten.mul.Tensor(div_49, sub_202);  div_49 = sub_202 = None
        mul_891 = torch.ops.aten.mul.Tensor(view_412, mul_114);  mul_114 = None
        sum_200 = torch.ops.aten.sum.dim_IntList(mul_891, [0, 1, 2]);  mul_891 = None
        sum_201 = torch.ops.aten.sum.dim_IntList(view_412, [0, 1, 2]);  view_412 = None
        add_472 = torch.ops.aten.add.Tensor(add_464, mul_890);  add_464 = mul_890 = None
        sum_202 = torch.ops.aten.sum.dim_IntList(add_472, [0, 1, 2], True)
        view_413 = torch.ops.aten.view.default(sum_202, [384]);  sum_202 = None
        view_414 = torch.ops.aten.view.default(add_472, [12544, 384])
        permute_521 = torch.ops.aten.permute.default(view_414, [1, 0])
        mm_205 = torch.ops.aten.mm.default(permute_521, view_41);  permute_521 = view_41 = None
        permute_522 = torch.ops.aten.permute.default(mm_205, [1, 0]);  mm_205 = None
        mm_206 = torch.ops.aten.mm.default(view_414, permute_523);  view_414 = permute_523 = None
        view_415 = torch.ops.aten.view.default(mm_206, [64, 14, 14, 384]);  mm_206 = None
        permute_524 = torch.ops.aten.permute.default(permute_522, [1, 0]);  permute_522 = None
        view_416 = torch.ops.aten.view.default(view_415, [64, 196, 12, 32]);  view_415 = None
        permute_525 = torch.ops.aten.permute.default(view_416, [0, 2, 1, 3]);  view_416 = None
        clone_169 = torch.ops.aten.clone.default(permute_525, memory_format = torch.contiguous_format);  permute_525 = None
        _unsafe_view_221 = torch.ops.aten._unsafe_view.default(clone_169, [768, 196, 32]);  clone_169 = None
        permute_526 = torch.ops.aten.permute.default(view_40, [0, 2, 1]);  view_40 = None
        bmm_92 = torch.ops.aten.bmm.default(permute_526, _unsafe_view_221);  permute_526 = None
        bmm_93 = torch.ops.aten.bmm.default(_unsafe_view_221, permute_527);  _unsafe_view_221 = permute_527 = None
        view_417 = torch.ops.aten.view.default(bmm_92, [64, 12, 196, 32]);  bmm_92 = None
        view_418 = torch.ops.aten.view.default(bmm_93, [64, 12, 196, 196]);  bmm_93 = None
        alias_129 = torch.ops.aten.alias.default(alias_58);  alias_58 = None
        alias_130 = torch.ops.aten.alias.default(alias_129);  alias_129 = None
        mul_892 = torch.ops.aten.mul.Tensor(view_418, alias_130);  view_418 = None
        sum_203 = torch.ops.aten.sum.dim_IntList(mul_892, [-1], True)
        mul_893 = torch.ops.aten.mul.Tensor(alias_130, sum_203);  alias_130 = sum_203 = None
        sub_203 = torch.ops.aten.sub.Tensor(mul_892, mul_893);  mul_892 = mul_893 = None
        mul_894 = torch.ops.aten.mul.Tensor(sub_203, 0.1767766952966369);  sub_203 = None
        view_419 = torch.ops.aten.view.default(mul_894, [768, 196, 196]);  mul_894 = None
        bmm_94 = torch.ops.aten.bmm.default(permute_528, view_419);  permute_528 = None
        bmm_95 = torch.ops.aten.bmm.default(view_419, permute_529);  view_419 = permute_529 = None
        view_420 = torch.ops.aten.view.default(bmm_94, [64, 12, 32, 196]);  bmm_94 = None
        view_421 = torch.ops.aten.view.default(bmm_95, [64, 12, 196, 32]);  bmm_95 = None
        permute_530 = torch.ops.aten.permute.default(view_420, [0, 1, 3, 2]);  view_420 = None
        cat_17 = torch.ops.aten.cat.default([view_421, permute_530, view_417]);  view_421 = permute_530 = view_417 = None
        view_422 = torch.ops.aten.view.default(cat_17, [3, 64, 12, 196, 32]);  cat_17 = None
        permute_531 = torch.ops.aten.permute.default(view_422, [1, 3, 0, 2, 4]);  view_422 = None
        clone_170 = torch.ops.aten.clone.default(permute_531, memory_format = torch.contiguous_format);  permute_531 = None
        _unsafe_view_222 = torch.ops.aten._unsafe_view.default(clone_170, [64, 14, 14, 1152]);  clone_170 = None
        view_423 = torch.ops.aten.view.default(_unsafe_view_222, [12544, 1152]);  _unsafe_view_222 = None
        permute_532 = torch.ops.aten.permute.default(view_423, [1, 0])
        mm_207 = torch.ops.aten.mm.default(permute_532, view_38);  permute_532 = view_38 = None
        permute_533 = torch.ops.aten.permute.default(mm_207, [1, 0]);  mm_207 = None
        mm_208 = torch.ops.aten.mm.default(view_423, permute_534);  view_423 = permute_534 = None
        view_424 = torch.ops.aten.view.default(mm_208, [64, 14, 14, 384]);  mm_208 = None
        permute_535 = torch.ops.aten.permute.default(permute_533, [1, 0]);  permute_533 = None
        mul_896 = torch.ops.aten.mul.Tensor(view_424, primals_79);  primals_79 = None
        mul_897 = torch.ops.aten.mul.Tensor(mul_896, 384)
        sum_204 = torch.ops.aten.sum.dim_IntList(mul_896, [3], True)
        mul_898 = torch.ops.aten.mul.Tensor(mul_896, mul_111);  mul_896 = None
        sum_205 = torch.ops.aten.sum.dim_IntList(mul_898, [3], True);  mul_898 = None
        mul_899 = torch.ops.aten.mul.Tensor(mul_111, sum_205);  sum_205 = None
        sub_205 = torch.ops.aten.sub.Tensor(mul_897, sum_204);  mul_897 = sum_204 = None
        sub_206 = torch.ops.aten.sub.Tensor(sub_205, mul_899);  sub_205 = mul_899 = None
        div_50 = torch.ops.aten.div.Tensor(reciprocal_18, 384);  reciprocal_18 = None
        mul_900 = torch.ops.aten.mul.Tensor(div_50, sub_206);  div_50 = sub_206 = None
        mul_901 = torch.ops.aten.mul.Tensor(view_424, mul_111);  mul_111 = None
        sum_206 = torch.ops.aten.sum.dim_IntList(mul_901, [0, 1, 2]);  mul_901 = None
        sum_207 = torch.ops.aten.sum.dim_IntList(view_424, [0, 1, 2]);  view_424 = None
        add_473 = torch.ops.aten.add.Tensor(add_472, mul_900);  add_472 = mul_900 = None
        sum_208 = torch.ops.aten.sum.dim_IntList(add_473, [0, 1, 2], True)
        view_425 = torch.ops.aten.view.default(sum_208, [384]);  sum_208 = None
        view_426 = torch.ops.aten.view.default(add_473, [12544, 384])
        permute_536 = torch.ops.aten.permute.default(view_426, [1, 0])
        mm_209 = torch.ops.aten.mm.default(permute_536, view_37);  permute_536 = view_37 = None
        permute_537 = torch.ops.aten.permute.default(mm_209, [1, 0]);  mm_209 = None
        mm_210 = torch.ops.aten.mm.default(view_426, permute_538);  view_426 = permute_538 = None
        view_427 = torch.ops.aten.view.default(mm_210, [64, 14, 14, 1152]);  mm_210 = None
        permute_539 = torch.ops.aten.permute.default(permute_537, [1, 0]);  permute_537 = None
        mul_918 = torch.ops.aten.mul.Tensor(view_427, add_480);  view_427 = add_480 = None
        sum_209 = torch.ops.aten.sum.dim_IntList(mul_918, [0, 1, 2], True)
        view_428 = torch.ops.aten.view.default(sum_209, [1152]);  sum_209 = None
        view_429 = torch.ops.aten.view.default(mul_918, [12544, 1152]);  mul_918 = None
        permute_540 = torch.ops.aten.permute.default(view_429, [1, 0])
        mm_211 = torch.ops.aten.mm.default(permute_540, view_36);  permute_540 = view_36 = None
        permute_541 = torch.ops.aten.permute.default(mm_211, [1, 0]);  mm_211 = None
        mm_212 = torch.ops.aten.mm.default(view_429, permute_542);  view_429 = permute_542 = None
        view_430 = torch.ops.aten.view.default(mm_212, [64, 14, 14, 384]);  mm_212 = None
        permute_543 = torch.ops.aten.permute.default(permute_541, [1, 0]);  permute_541 = None
        mul_920 = torch.ops.aten.mul.Tensor(view_430, primals_73);  primals_73 = None
        mul_921 = torch.ops.aten.mul.Tensor(mul_920, 384)
        sum_210 = torch.ops.aten.sum.dim_IntList(mul_920, [3], True)
        mul_922 = torch.ops.aten.mul.Tensor(mul_920, mul_96);  mul_920 = None
        sum_211 = torch.ops.aten.sum.dim_IntList(mul_922, [3], True);  mul_922 = None
        mul_923 = torch.ops.aten.mul.Tensor(mul_96, sum_211);  sum_211 = None
        sub_209 = torch.ops.aten.sub.Tensor(mul_921, sum_210);  mul_921 = sum_210 = None
        sub_210 = torch.ops.aten.sub.Tensor(sub_209, mul_923);  sub_209 = mul_923 = None
        div_51 = torch.ops.aten.div.Tensor(reciprocal_16, 384);  reciprocal_16 = None
        mul_924 = torch.ops.aten.mul.Tensor(div_51, sub_210);  div_51 = sub_210 = None
        mul_925 = torch.ops.aten.mul.Tensor(view_430, mul_96);  mul_96 = None
        sum_212 = torch.ops.aten.sum.dim_IntList(mul_925, [0, 1, 2]);  mul_925 = None
        sum_213 = torch.ops.aten.sum.dim_IntList(view_430, [0, 1, 2]);  view_430 = None
        add_481 = torch.ops.aten.add.Tensor(add_473, mul_924);  add_473 = mul_924 = None
        sum_214 = torch.ops.aten.sum.dim_IntList(add_481, [0, 1, 2], True)
        view_431 = torch.ops.aten.view.default(sum_214, [384]);  sum_214 = None
        view_432 = torch.ops.aten.view.default(add_481, [12544, 384])
        permute_544 = torch.ops.aten.permute.default(view_432, [1, 0])
        mm_213 = torch.ops.aten.mm.default(permute_544, view_35);  permute_544 = view_35 = None
        permute_545 = torch.ops.aten.permute.default(mm_213, [1, 0]);  mm_213 = None
        mm_214 = torch.ops.aten.mm.default(view_432, permute_546);  view_432 = permute_546 = None
        view_433 = torch.ops.aten.view.default(mm_214, [64, 14, 14, 384]);  mm_214 = None
        permute_547 = torch.ops.aten.permute.default(permute_545, [1, 0]);  permute_545 = None
        view_434 = torch.ops.aten.view.default(view_433, [64, 196, 12, 32]);  view_433 = None
        permute_548 = torch.ops.aten.permute.default(view_434, [0, 2, 1, 3]);  view_434 = None
        clone_173 = torch.ops.aten.clone.default(permute_548, memory_format = torch.contiguous_format);  permute_548 = None
        _unsafe_view_223 = torch.ops.aten._unsafe_view.default(clone_173, [768, 196, 32]);  clone_173 = None
        permute_549 = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
        bmm_96 = torch.ops.aten.bmm.default(permute_549, _unsafe_view_223);  permute_549 = None
        bmm_97 = torch.ops.aten.bmm.default(_unsafe_view_223, permute_550);  _unsafe_view_223 = permute_550 = None
        view_435 = torch.ops.aten.view.default(bmm_96, [64, 12, 196, 32]);  bmm_96 = None
        view_436 = torch.ops.aten.view.default(bmm_97, [64, 12, 196, 196]);  bmm_97 = None
        alias_131 = torch.ops.aten.alias.default(alias_55);  alias_55 = None
        alias_132 = torch.ops.aten.alias.default(alias_131);  alias_131 = None
        mul_926 = torch.ops.aten.mul.Tensor(view_436, alias_132);  view_436 = None
        sum_215 = torch.ops.aten.sum.dim_IntList(mul_926, [-1], True)
        mul_927 = torch.ops.aten.mul.Tensor(alias_132, sum_215);  alias_132 = sum_215 = None
        sub_211 = torch.ops.aten.sub.Tensor(mul_926, mul_927);  mul_926 = mul_927 = None
        mul_928 = torch.ops.aten.mul.Tensor(sub_211, 0.1767766952966369);  sub_211 = None
        view_437 = torch.ops.aten.view.default(mul_928, [768, 196, 196]);  mul_928 = None
        bmm_98 = torch.ops.aten.bmm.default(permute_551, view_437);  permute_551 = None
        bmm_99 = torch.ops.aten.bmm.default(view_437, permute_552);  view_437 = permute_552 = None
        view_438 = torch.ops.aten.view.default(bmm_98, [64, 12, 32, 196]);  bmm_98 = None
        view_439 = torch.ops.aten.view.default(bmm_99, [64, 12, 196, 32]);  bmm_99 = None
        permute_553 = torch.ops.aten.permute.default(view_438, [0, 1, 3, 2]);  view_438 = None
        cat_18 = torch.ops.aten.cat.default([view_439, permute_553, view_435]);  view_439 = permute_553 = view_435 = None
        view_440 = torch.ops.aten.view.default(cat_18, [3, 64, 12, 196, 32]);  cat_18 = None
        permute_554 = torch.ops.aten.permute.default(view_440, [1, 3, 0, 2, 4]);  view_440 = None
        clone_174 = torch.ops.aten.clone.default(permute_554, memory_format = torch.contiguous_format);  permute_554 = None
        _unsafe_view_224 = torch.ops.aten._unsafe_view.default(clone_174, [64, 14, 14, 1152]);  clone_174 = None
        view_441 = torch.ops.aten.view.default(_unsafe_view_224, [12544, 1152]);  _unsafe_view_224 = None
        permute_555 = torch.ops.aten.permute.default(view_441, [1, 0])
        mm_215 = torch.ops.aten.mm.default(permute_555, view_32);  permute_555 = view_32 = None
        permute_556 = torch.ops.aten.permute.default(mm_215, [1, 0]);  mm_215 = None
        mm_216 = torch.ops.aten.mm.default(view_441, permute_557);  view_441 = permute_557 = None
        view_442 = torch.ops.aten.view.default(mm_216, [64, 14, 14, 384]);  mm_216 = None
        permute_558 = torch.ops.aten.permute.default(permute_556, [1, 0]);  permute_556 = None
        mul_930 = torch.ops.aten.mul.Tensor(view_442, primals_68);  primals_68 = None
        mul_931 = torch.ops.aten.mul.Tensor(mul_930, 384)
        sum_216 = torch.ops.aten.sum.dim_IntList(mul_930, [3], True)
        mul_932 = torch.ops.aten.mul.Tensor(mul_930, mul_93);  mul_930 = None
        sum_217 = torch.ops.aten.sum.dim_IntList(mul_932, [3], True);  mul_932 = None
        mul_933 = torch.ops.aten.mul.Tensor(mul_93, sum_217);  sum_217 = None
        sub_213 = torch.ops.aten.sub.Tensor(mul_931, sum_216);  mul_931 = sum_216 = None
        sub_214 = torch.ops.aten.sub.Tensor(sub_213, mul_933);  sub_213 = mul_933 = None
        div_52 = torch.ops.aten.div.Tensor(reciprocal_15, 384);  reciprocal_15 = None
        mul_934 = torch.ops.aten.mul.Tensor(div_52, sub_214);  div_52 = sub_214 = None
        mul_935 = torch.ops.aten.mul.Tensor(view_442, mul_93);  mul_93 = None
        sum_218 = torch.ops.aten.sum.dim_IntList(mul_935, [0, 1, 2]);  mul_935 = None
        sum_219 = torch.ops.aten.sum.dim_IntList(view_442, [0, 1, 2]);  view_442 = None
        add_482 = torch.ops.aten.add.Tensor(add_481, mul_934);  add_481 = mul_934 = None
        sum_220 = torch.ops.aten.sum.dim_IntList(add_482, [0], True)
        permute_559 = torch.ops.aten.permute.default(add_482, [0, 3, 1, 2]);  add_482 = None
        convolution_backward = torch.ops.aten.convolution_backward.default(permute_559, permute_57, primals_66, [384], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  permute_559 = permute_57 = primals_66 = None
        getitem_130 = convolution_backward[0]
        getitem_131 = convolution_backward[1]
        getitem_132 = convolution_backward[2];  convolution_backward = None
        permute_560 = torch.ops.aten.permute.default(getitem_130, [0, 2, 3, 1]);  getitem_130 = None
        sum_221 = torch.ops.aten.sum.dim_IntList(permute_560, [0, 1, 2], True)
        view_443 = torch.ops.aten.view.default(sum_221, [192]);  sum_221 = None
        clone_176 = torch.ops.aten.clone.default(permute_560, memory_format = torch.contiguous_format)
        _unsafe_view_225 = torch.ops.aten._unsafe_view.default(clone_176, [50176, 192]);  clone_176 = None
        permute_561 = torch.ops.aten.permute.default(_unsafe_view_225, [1, 0])
        mm_217 = torch.ops.aten.mm.default(permute_561, view_31);  permute_561 = view_31 = None
        permute_562 = torch.ops.aten.permute.default(mm_217, [1, 0]);  mm_217 = None
        mm_218 = torch.ops.aten.mm.default(_unsafe_view_225, permute_563);  _unsafe_view_225 = permute_563 = None
        view_444 = torch.ops.aten.view.default(mm_218, [64, 28, 28, 576]);  mm_218 = None
        permute_564 = torch.ops.aten.permute.default(permute_562, [1, 0]);  permute_562 = None
        mul_952 = torch.ops.aten.mul.Tensor(view_444, add_489);  view_444 = add_489 = None
        sum_222 = torch.ops.aten.sum.dim_IntList(mul_952, [0, 1, 2], True)
        view_445 = torch.ops.aten.view.default(sum_222, [576]);  sum_222 = None
        view_446 = torch.ops.aten.view.default(mul_952, [50176, 576]);  mul_952 = None
        permute_565 = torch.ops.aten.permute.default(view_446, [1, 0])
        mm_219 = torch.ops.aten.mm.default(permute_565, view_30);  permute_565 = view_30 = None
        permute_566 = torch.ops.aten.permute.default(mm_219, [1, 0]);  mm_219 = None
        mm_220 = torch.ops.aten.mm.default(view_446, permute_567);  view_446 = permute_567 = None
        view_447 = torch.ops.aten.view.default(mm_220, [64, 28, 28, 192]);  mm_220 = None
        permute_568 = torch.ops.aten.permute.default(permute_566, [1, 0]);  permute_566 = None
        mul_954 = torch.ops.aten.mul.Tensor(view_447, primals_60);  primals_60 = None
        mul_955 = torch.ops.aten.mul.Tensor(mul_954, 192)
        sum_223 = torch.ops.aten.sum.dim_IntList(mul_954, [3], True)
        mul_956 = torch.ops.aten.mul.Tensor(mul_954, mul_78);  mul_954 = None
        sum_224 = torch.ops.aten.sum.dim_IntList(mul_956, [3], True);  mul_956 = None
        mul_957 = torch.ops.aten.mul.Tensor(mul_78, sum_224);  sum_224 = None
        sub_217 = torch.ops.aten.sub.Tensor(mul_955, sum_223);  mul_955 = sum_223 = None
        sub_218 = torch.ops.aten.sub.Tensor(sub_217, mul_957);  sub_217 = mul_957 = None
        div_53 = torch.ops.aten.div.Tensor(reciprocal_13, 192);  reciprocal_13 = None
        mul_958 = torch.ops.aten.mul.Tensor(div_53, sub_218);  div_53 = sub_218 = None
        mul_959 = torch.ops.aten.mul.Tensor(view_447, mul_78);  mul_78 = None
        sum_225 = torch.ops.aten.sum.dim_IntList(mul_959, [0, 1, 2]);  mul_959 = None
        sum_226 = torch.ops.aten.sum.dim_IntList(view_447, [0, 1, 2]);  view_447 = None
        add_490 = torch.ops.aten.add.Tensor(permute_560, mul_958);  permute_560 = mul_958 = None
        sum_227 = torch.ops.aten.sum.dim_IntList(add_490, [0, 1, 2], True)
        view_448 = torch.ops.aten.view.default(sum_227, [192]);  sum_227 = None
        clone_178 = torch.ops.aten.clone.default(add_490, memory_format = torch.contiguous_format)
        _unsafe_view_226 = torch.ops.aten._unsafe_view.default(clone_178, [50176, 192]);  clone_178 = None
        permute_569 = torch.ops.aten.permute.default(_unsafe_view_226, [1, 0])
        mm_221 = torch.ops.aten.mm.default(permute_569, _unsafe_view_36);  permute_569 = _unsafe_view_36 = None
        permute_570 = torch.ops.aten.permute.default(mm_221, [1, 0]);  mm_221 = None
        mm_222 = torch.ops.aten.mm.default(_unsafe_view_226, permute_571);  _unsafe_view_226 = permute_571 = None
        view_449 = torch.ops.aten.view.default(mm_222, [64, 28, 28, 192]);  mm_222 = None
        permute_572 = torch.ops.aten.permute.default(permute_570, [1, 0]);  permute_570 = None
        permute_573 = torch.ops.aten.permute.default(view_449, [0, 3, 1, 2]);  view_449 = None
        col2im_backward = torch.ops.aten.col2im_backward.default(permute_573, [3, 3], [1, 1], [1, 1], [2, 2]);  permute_573 = None
        view_450 = torch.ops.aten.view.default(col2im_backward, [64, 6, 32, 9, 196]);  col2im_backward = None
        permute_574 = torch.ops.aten.permute.default(view_450, [0, 1, 4, 3, 2]);  view_450 = None
        clone_179 = torch.ops.aten.clone.default(permute_574, memory_format = torch.contiguous_format);  permute_574 = None
        _unsafe_view_227 = torch.ops.aten._unsafe_view.default(clone_179, [75264, 9, 32]);  clone_179 = None
        permute_575 = torch.ops.aten.permute.default(view_28, [0, 2, 1]);  view_28 = None
        bmm_100 = torch.ops.aten.bmm.default(permute_575, _unsafe_view_227);  permute_575 = None
        bmm_101 = torch.ops.aten.bmm.default(_unsafe_view_227, permute_576);  _unsafe_view_227 = permute_576 = None
        view_451 = torch.ops.aten.view.default(bmm_100, [64, 6, 196, 9, 32]);  bmm_100 = None
        view_452 = torch.ops.aten.view.default(bmm_101, [64, 6, 196, 9, 9]);  bmm_101 = None
        alias_133 = torch.ops.aten.alias.default(alias_48);  alias_48 = None
        alias_134 = torch.ops.aten.alias.default(alias_133);  alias_133 = None
        mul_960 = torch.ops.aten.mul.Tensor(view_452, alias_134);  view_452 = None
        sum_228 = torch.ops.aten.sum.dim_IntList(mul_960, [-1], True)
        mul_961 = torch.ops.aten.mul.Tensor(alias_134, sum_228);  alias_134 = sum_228 = None
        sub_219 = torch.ops.aten.sub.Tensor(mul_960, mul_961);  mul_960 = mul_961 = None
        mul_962 = torch.ops.aten.mul.Tensor(sub_219, 0.1767766952966369);  sub_219 = None
        permute_577 = torch.ops.aten.permute.default(mul_962, [0, 2, 1, 3, 4]);  mul_962 = None
        clone_180 = torch.ops.aten.clone.default(permute_577, memory_format = torch.contiguous_format);  permute_577 = None
        _unsafe_view_228 = torch.ops.aten._unsafe_view.default(clone_180, [64, 14, 14, 486]);  clone_180 = None
        sum_229 = torch.ops.aten.sum.dim_IntList(_unsafe_view_228, [0, 1, 2], True)
        view_453 = torch.ops.aten.view.default(sum_229, [486]);  sum_229 = None
        view_454 = torch.ops.aten.view.default(_unsafe_view_228, [12544, 486]);  _unsafe_view_228 = None
        permute_578 = torch.ops.aten.permute.default(view_454, [1, 0])
        mm_223 = torch.ops.aten.mm.default(permute_578, view_26);  permute_578 = view_26 = None
        permute_579 = torch.ops.aten.permute.default(mm_223, [1, 0]);  mm_223 = None
        mm_224 = torch.ops.aten.mm.default(view_454, permute_580);  view_454 = permute_580 = None
        view_455 = torch.ops.aten.view.default(mm_224, [64, 14, 14, 192]);  mm_224 = None
        permute_581 = torch.ops.aten.permute.default(permute_579, [1, 0]);  permute_579 = None
        permute_582 = torch.ops.aten.permute.default(view_455, [0, 3, 1, 2]);  view_455 = None
        avg_pool2d_backward = torch.ops.aten.avg_pool2d_backward.default(permute_582, permute_47, [2, 2], [2, 2], [0, 0], True, True, None);  permute_582 = permute_47 = None
        permute_583 = torch.ops.aten.permute.default(avg_pool2d_backward, [0, 2, 3, 1]);  avg_pool2d_backward = None
        permute_584 = torch.ops.aten.permute.default(view_451, [0, 1, 4, 3, 2]);  view_451 = None
        clone_181 = torch.ops.aten.clone.default(permute_584, memory_format = torch.contiguous_format);  permute_584 = None
        _unsafe_view_229 = torch.ops.aten._unsafe_view.default(clone_181, [64, 1728, 196]);  clone_181 = None
        im2col_backward = torch.ops.aten.im2col_backward.default(_unsafe_view_229, [28, 28], [3, 3], [1, 1], [1, 1], [2, 2]);  _unsafe_view_229 = None
        permute_585 = torch.ops.aten.permute.default(im2col_backward, [0, 2, 3, 1]);  im2col_backward = None
        clone_182 = torch.ops.aten.clone.default(permute_585, memory_format = torch.contiguous_format);  permute_585 = None
        _unsafe_view_230 = torch.ops.aten._unsafe_view.default(clone_182, [50176, 192]);  clone_182 = None
        permute_586 = torch.ops.aten.permute.default(_unsafe_view_230, [1, 0])
        mm_225 = torch.ops.aten.mm.default(permute_586, view_24);  permute_586 = view_24 = None
        permute_587 = torch.ops.aten.permute.default(mm_225, [1, 0]);  mm_225 = None
        mm_226 = torch.ops.aten.mm.default(_unsafe_view_230, permute_588);  _unsafe_view_230 = permute_588 = None
        view_456 = torch.ops.aten.view.default(mm_226, [64, 28, 28, 192]);  mm_226 = None
        add_491 = torch.ops.aten.add.Tensor(permute_583, view_456);  permute_583 = view_456 = None
        permute_589 = torch.ops.aten.permute.default(permute_587, [1, 0]);  permute_587 = None
        mul_964 = torch.ops.aten.mul.Tensor(add_491, primals_53);  primals_53 = None
        mul_965 = torch.ops.aten.mul.Tensor(mul_964, 192)
        sum_230 = torch.ops.aten.sum.dim_IntList(mul_964, [3], True)
        mul_966 = torch.ops.aten.mul.Tensor(mul_964, mul_75);  mul_964 = None
        sum_231 = torch.ops.aten.sum.dim_IntList(mul_966, [3], True);  mul_966 = None
        mul_967 = torch.ops.aten.mul.Tensor(mul_75, sum_231);  sum_231 = None
        sub_221 = torch.ops.aten.sub.Tensor(mul_965, sum_230);  mul_965 = sum_230 = None
        sub_222 = torch.ops.aten.sub.Tensor(sub_221, mul_967);  sub_221 = mul_967 = None
        div_54 = torch.ops.aten.div.Tensor(reciprocal_12, 192);  reciprocal_12 = None
        mul_968 = torch.ops.aten.mul.Tensor(div_54, sub_222);  div_54 = sub_222 = None
        mul_969 = torch.ops.aten.mul.Tensor(add_491, mul_75);  mul_75 = None
        sum_232 = torch.ops.aten.sum.dim_IntList(mul_969, [0, 1, 2]);  mul_969 = None
        sum_233 = torch.ops.aten.sum.dim_IntList(add_491, [0, 1, 2]);  add_491 = None
        add_492 = torch.ops.aten.add.Tensor(add_490, mul_968);  add_490 = mul_968 = None
        sum_234 = torch.ops.aten.sum.dim_IntList(add_492, [0, 1, 2], True)
        view_457 = torch.ops.aten.view.default(sum_234, [192]);  sum_234 = None
        clone_184 = torch.ops.aten.clone.default(add_492, memory_format = torch.contiguous_format)
        _unsafe_view_231 = torch.ops.aten._unsafe_view.default(clone_184, [50176, 192]);  clone_184 = None
        permute_590 = torch.ops.aten.permute.default(_unsafe_view_231, [1, 0])
        mm_227 = torch.ops.aten.mm.default(permute_590, view_23);  permute_590 = view_23 = None
        permute_591 = torch.ops.aten.permute.default(mm_227, [1, 0]);  mm_227 = None
        mm_228 = torch.ops.aten.mm.default(_unsafe_view_231, permute_592);  _unsafe_view_231 = permute_592 = None
        view_458 = torch.ops.aten.view.default(mm_228, [64, 28, 28, 576]);  mm_228 = None
        permute_593 = torch.ops.aten.permute.default(permute_591, [1, 0]);  permute_591 = None
        mul_986 = torch.ops.aten.mul.Tensor(view_458, add_499);  view_458 = add_499 = None
        sum_235 = torch.ops.aten.sum.dim_IntList(mul_986, [0, 1, 2], True)
        view_459 = torch.ops.aten.view.default(sum_235, [576]);  sum_235 = None
        view_460 = torch.ops.aten.view.default(mul_986, [50176, 576]);  mul_986 = None
        permute_594 = torch.ops.aten.permute.default(view_460, [1, 0])
        mm_229 = torch.ops.aten.mm.default(permute_594, view_22);  permute_594 = view_22 = None
        permute_595 = torch.ops.aten.permute.default(mm_229, [1, 0]);  mm_229 = None
        mm_230 = torch.ops.aten.mm.default(view_460, permute_596);  view_460 = permute_596 = None
        view_461 = torch.ops.aten.view.default(mm_230, [64, 28, 28, 192]);  mm_230 = None
        permute_597 = torch.ops.aten.permute.default(permute_595, [1, 0]);  permute_595 = None
        mul_988 = torch.ops.aten.mul.Tensor(view_461, primals_47);  primals_47 = None
        mul_989 = torch.ops.aten.mul.Tensor(mul_988, 192)
        sum_236 = torch.ops.aten.sum.dim_IntList(mul_988, [3], True)
        mul_990 = torch.ops.aten.mul.Tensor(mul_988, mul_60);  mul_988 = None
        sum_237 = torch.ops.aten.sum.dim_IntList(mul_990, [3], True);  mul_990 = None
        mul_991 = torch.ops.aten.mul.Tensor(mul_60, sum_237);  sum_237 = None
        sub_225 = torch.ops.aten.sub.Tensor(mul_989, sum_236);  mul_989 = sum_236 = None
        sub_226 = torch.ops.aten.sub.Tensor(sub_225, mul_991);  sub_225 = mul_991 = None
        div_55 = torch.ops.aten.div.Tensor(reciprocal_10, 192);  reciprocal_10 = None
        mul_992 = torch.ops.aten.mul.Tensor(div_55, sub_226);  div_55 = sub_226 = None
        mul_993 = torch.ops.aten.mul.Tensor(view_461, mul_60);  mul_60 = None
        sum_238 = torch.ops.aten.sum.dim_IntList(mul_993, [0, 1, 2]);  mul_993 = None
        sum_239 = torch.ops.aten.sum.dim_IntList(view_461, [0, 1, 2]);  view_461 = None
        add_500 = torch.ops.aten.add.Tensor(add_492, mul_992);  add_492 = mul_992 = None
        sum_240 = torch.ops.aten.sum.dim_IntList(add_500, [0, 1, 2], True)
        view_462 = torch.ops.aten.view.default(sum_240, [192]);  sum_240 = None
        clone_186 = torch.ops.aten.clone.default(add_500, memory_format = torch.contiguous_format)
        _unsafe_view_232 = torch.ops.aten._unsafe_view.default(clone_186, [50176, 192]);  clone_186 = None
        permute_598 = torch.ops.aten.permute.default(_unsafe_view_232, [1, 0])
        mm_231 = torch.ops.aten.mm.default(permute_598, _unsafe_view_26);  permute_598 = _unsafe_view_26 = None
        permute_599 = torch.ops.aten.permute.default(mm_231, [1, 0]);  mm_231 = None
        mm_232 = torch.ops.aten.mm.default(_unsafe_view_232, permute_600);  _unsafe_view_232 = permute_600 = None
        view_463 = torch.ops.aten.view.default(mm_232, [64, 28, 28, 192]);  mm_232 = None
        permute_601 = torch.ops.aten.permute.default(permute_599, [1, 0]);  permute_599 = None
        permute_602 = torch.ops.aten.permute.default(view_463, [0, 3, 1, 2]);  view_463 = None
        col2im_backward_1 = torch.ops.aten.col2im_backward.default(permute_602, [3, 3], [1, 1], [1, 1], [2, 2]);  permute_602 = None
        view_464 = torch.ops.aten.view.default(col2im_backward_1, [64, 6, 32, 9, 196]);  col2im_backward_1 = None
        permute_603 = torch.ops.aten.permute.default(view_464, [0, 1, 4, 3, 2]);  view_464 = None
        clone_187 = torch.ops.aten.clone.default(permute_603, memory_format = torch.contiguous_format);  permute_603 = None
        _unsafe_view_233 = torch.ops.aten._unsafe_view.default(clone_187, [75264, 9, 32]);  clone_187 = None
        permute_604 = torch.ops.aten.permute.default(view_20, [0, 2, 1]);  view_20 = None
        bmm_102 = torch.ops.aten.bmm.default(permute_604, _unsafe_view_233);  permute_604 = None
        bmm_103 = torch.ops.aten.bmm.default(_unsafe_view_233, permute_605);  _unsafe_view_233 = permute_605 = None
        view_465 = torch.ops.aten.view.default(bmm_102, [64, 6, 196, 9, 32]);  bmm_102 = None
        view_466 = torch.ops.aten.view.default(bmm_103, [64, 6, 196, 9, 9]);  bmm_103 = None
        alias_135 = torch.ops.aten.alias.default(alias_37);  alias_37 = None
        alias_136 = torch.ops.aten.alias.default(alias_135);  alias_135 = None
        mul_994 = torch.ops.aten.mul.Tensor(view_466, alias_136);  view_466 = None
        sum_241 = torch.ops.aten.sum.dim_IntList(mul_994, [-1], True)
        mul_995 = torch.ops.aten.mul.Tensor(alias_136, sum_241);  alias_136 = sum_241 = None
        sub_227 = torch.ops.aten.sub.Tensor(mul_994, mul_995);  mul_994 = mul_995 = None
        mul_996 = torch.ops.aten.mul.Tensor(sub_227, 0.1767766952966369);  sub_227 = None
        permute_606 = torch.ops.aten.permute.default(mul_996, [0, 2, 1, 3, 4]);  mul_996 = None
        clone_188 = torch.ops.aten.clone.default(permute_606, memory_format = torch.contiguous_format);  permute_606 = None
        _unsafe_view_234 = torch.ops.aten._unsafe_view.default(clone_188, [64, 14, 14, 486]);  clone_188 = None
        sum_242 = torch.ops.aten.sum.dim_IntList(_unsafe_view_234, [0, 1, 2], True)
        view_467 = torch.ops.aten.view.default(sum_242, [486]);  sum_242 = None
        view_468 = torch.ops.aten.view.default(_unsafe_view_234, [12544, 486]);  _unsafe_view_234 = None
        permute_607 = torch.ops.aten.permute.default(view_468, [1, 0])
        mm_233 = torch.ops.aten.mm.default(permute_607, view_18);  permute_607 = view_18 = None
        permute_608 = torch.ops.aten.permute.default(mm_233, [1, 0]);  mm_233 = None
        mm_234 = torch.ops.aten.mm.default(view_468, permute_609);  view_468 = permute_609 = None
        view_469 = torch.ops.aten.view.default(mm_234, [64, 14, 14, 192]);  mm_234 = None
        permute_610 = torch.ops.aten.permute.default(permute_608, [1, 0]);  permute_608 = None
        permute_611 = torch.ops.aten.permute.default(view_469, [0, 3, 1, 2]);  view_469 = None
        avg_pool2d_backward_1 = torch.ops.aten.avg_pool2d_backward.default(permute_611, permute_33, [2, 2], [2, 2], [0, 0], True, True, None);  permute_611 = permute_33 = None
        permute_612 = torch.ops.aten.permute.default(avg_pool2d_backward_1, [0, 2, 3, 1]);  avg_pool2d_backward_1 = None
        permute_613 = torch.ops.aten.permute.default(view_465, [0, 1, 4, 3, 2]);  view_465 = None
        clone_189 = torch.ops.aten.clone.default(permute_613, memory_format = torch.contiguous_format);  permute_613 = None
        _unsafe_view_235 = torch.ops.aten._unsafe_view.default(clone_189, [64, 1728, 196]);  clone_189 = None
        im2col_backward_1 = torch.ops.aten.im2col_backward.default(_unsafe_view_235, [28, 28], [3, 3], [1, 1], [1, 1], [2, 2]);  _unsafe_view_235 = None
        permute_614 = torch.ops.aten.permute.default(im2col_backward_1, [0, 2, 3, 1]);  im2col_backward_1 = None
        clone_190 = torch.ops.aten.clone.default(permute_614, memory_format = torch.contiguous_format);  permute_614 = None
        _unsafe_view_236 = torch.ops.aten._unsafe_view.default(clone_190, [50176, 192]);  clone_190 = None
        permute_615 = torch.ops.aten.permute.default(_unsafe_view_236, [1, 0])
        mm_235 = torch.ops.aten.mm.default(permute_615, view_16);  permute_615 = view_16 = None
        permute_616 = torch.ops.aten.permute.default(mm_235, [1, 0]);  mm_235 = None
        mm_236 = torch.ops.aten.mm.default(_unsafe_view_236, permute_617);  _unsafe_view_236 = permute_617 = None
        view_470 = torch.ops.aten.view.default(mm_236, [64, 28, 28, 192]);  mm_236 = None
        add_501 = torch.ops.aten.add.Tensor(permute_612, view_470);  permute_612 = view_470 = None
        permute_618 = torch.ops.aten.permute.default(permute_616, [1, 0]);  permute_616 = None
        mul_998 = torch.ops.aten.mul.Tensor(add_501, primals_40);  primals_40 = None
        mul_999 = torch.ops.aten.mul.Tensor(mul_998, 192)
        sum_243 = torch.ops.aten.sum.dim_IntList(mul_998, [3], True)
        mul_1000 = torch.ops.aten.mul.Tensor(mul_998, mul_57);  mul_998 = None
        sum_244 = torch.ops.aten.sum.dim_IntList(mul_1000, [3], True);  mul_1000 = None
        mul_1001 = torch.ops.aten.mul.Tensor(mul_57, sum_244);  sum_244 = None
        sub_229 = torch.ops.aten.sub.Tensor(mul_999, sum_243);  mul_999 = sum_243 = None
        sub_230 = torch.ops.aten.sub.Tensor(sub_229, mul_1001);  sub_229 = mul_1001 = None
        div_56 = torch.ops.aten.div.Tensor(reciprocal_9, 192);  reciprocal_9 = None
        mul_1002 = torch.ops.aten.mul.Tensor(div_56, sub_230);  div_56 = sub_230 = None
        mul_1003 = torch.ops.aten.mul.Tensor(add_501, mul_57);  mul_57 = None
        sum_245 = torch.ops.aten.sum.dim_IntList(mul_1003, [0, 1, 2]);  mul_1003 = None
        sum_246 = torch.ops.aten.sum.dim_IntList(add_501, [0, 1, 2]);  add_501 = None
        add_502 = torch.ops.aten.add.Tensor(add_500, mul_1002);  add_500 = mul_1002 = None
        sum_247 = torch.ops.aten.sum.dim_IntList(add_502, [0, 1, 2], True)
        view_471 = torch.ops.aten.view.default(sum_247, [192]);  sum_247 = None
        clone_192 = torch.ops.aten.clone.default(add_502, memory_format = torch.contiguous_format)
        _unsafe_view_237 = torch.ops.aten._unsafe_view.default(clone_192, [50176, 192]);  clone_192 = None
        permute_619 = torch.ops.aten.permute.default(_unsafe_view_237, [1, 0])
        mm_237 = torch.ops.aten.mm.default(permute_619, view_15);  permute_619 = view_15 = None
        permute_620 = torch.ops.aten.permute.default(mm_237, [1, 0]);  mm_237 = None
        mm_238 = torch.ops.aten.mm.default(_unsafe_view_237, permute_621);  _unsafe_view_237 = permute_621 = None
        view_472 = torch.ops.aten.view.default(mm_238, [64, 28, 28, 576]);  mm_238 = None
        permute_622 = torch.ops.aten.permute.default(permute_620, [1, 0]);  permute_620 = None
        mul_1020 = torch.ops.aten.mul.Tensor(view_472, add_509);  view_472 = add_509 = None
        sum_248 = torch.ops.aten.sum.dim_IntList(mul_1020, [0, 1, 2], True)
        view_473 = torch.ops.aten.view.default(sum_248, [576]);  sum_248 = None
        view_474 = torch.ops.aten.view.default(mul_1020, [50176, 576]);  mul_1020 = None
        permute_623 = torch.ops.aten.permute.default(view_474, [1, 0])
        mm_239 = torch.ops.aten.mm.default(permute_623, view_14);  permute_623 = view_14 = None
        permute_624 = torch.ops.aten.permute.default(mm_239, [1, 0]);  mm_239 = None
        mm_240 = torch.ops.aten.mm.default(view_474, permute_625);  view_474 = permute_625 = None
        view_475 = torch.ops.aten.view.default(mm_240, [64, 28, 28, 192]);  mm_240 = None
        permute_626 = torch.ops.aten.permute.default(permute_624, [1, 0]);  permute_624 = None
        mul_1022 = torch.ops.aten.mul.Tensor(view_475, primals_34);  primals_34 = None
        mul_1023 = torch.ops.aten.mul.Tensor(mul_1022, 192)
        sum_249 = torch.ops.aten.sum.dim_IntList(mul_1022, [3], True)
        mul_1024 = torch.ops.aten.mul.Tensor(mul_1022, mul_42);  mul_1022 = None
        sum_250 = torch.ops.aten.sum.dim_IntList(mul_1024, [3], True);  mul_1024 = None
        mul_1025 = torch.ops.aten.mul.Tensor(mul_42, sum_250);  sum_250 = None
        sub_233 = torch.ops.aten.sub.Tensor(mul_1023, sum_249);  mul_1023 = sum_249 = None
        sub_234 = torch.ops.aten.sub.Tensor(sub_233, mul_1025);  sub_233 = mul_1025 = None
        div_57 = torch.ops.aten.div.Tensor(reciprocal_7, 192);  reciprocal_7 = None
        mul_1026 = torch.ops.aten.mul.Tensor(div_57, sub_234);  div_57 = sub_234 = None
        mul_1027 = torch.ops.aten.mul.Tensor(view_475, mul_42);  mul_42 = None
        sum_251 = torch.ops.aten.sum.dim_IntList(mul_1027, [0, 1, 2]);  mul_1027 = None
        sum_252 = torch.ops.aten.sum.dim_IntList(view_475, [0, 1, 2]);  view_475 = None
        add_510 = torch.ops.aten.add.Tensor(add_502, mul_1026);  add_502 = mul_1026 = None
        sum_253 = torch.ops.aten.sum.dim_IntList(add_510, [0, 1, 2], True)
        view_476 = torch.ops.aten.view.default(sum_253, [192]);  sum_253 = None
        clone_194 = torch.ops.aten.clone.default(add_510, memory_format = torch.contiguous_format)
        _unsafe_view_238 = torch.ops.aten._unsafe_view.default(clone_194, [50176, 192]);  clone_194 = None
        permute_627 = torch.ops.aten.permute.default(_unsafe_view_238, [1, 0])
        mm_241 = torch.ops.aten.mm.default(permute_627, _unsafe_view_16);  permute_627 = _unsafe_view_16 = None
        permute_628 = torch.ops.aten.permute.default(mm_241, [1, 0]);  mm_241 = None
        mm_242 = torch.ops.aten.mm.default(_unsafe_view_238, permute_629);  _unsafe_view_238 = permute_629 = None
        view_477 = torch.ops.aten.view.default(mm_242, [64, 28, 28, 192]);  mm_242 = None
        permute_630 = torch.ops.aten.permute.default(permute_628, [1, 0]);  permute_628 = None
        permute_631 = torch.ops.aten.permute.default(view_477, [0, 3, 1, 2]);  view_477 = None
        col2im_backward_2 = torch.ops.aten.col2im_backward.default(permute_631, [3, 3], [1, 1], [1, 1], [2, 2]);  permute_631 = None
        view_478 = torch.ops.aten.view.default(col2im_backward_2, [64, 6, 32, 9, 196]);  col2im_backward_2 = None
        permute_632 = torch.ops.aten.permute.default(view_478, [0, 1, 4, 3, 2]);  view_478 = None
        clone_195 = torch.ops.aten.clone.default(permute_632, memory_format = torch.contiguous_format);  permute_632 = None
        _unsafe_view_239 = torch.ops.aten._unsafe_view.default(clone_195, [75264, 9, 32]);  clone_195 = None
        permute_633 = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        bmm_104 = torch.ops.aten.bmm.default(permute_633, _unsafe_view_239);  permute_633 = None
        bmm_105 = torch.ops.aten.bmm.default(_unsafe_view_239, permute_634);  _unsafe_view_239 = permute_634 = None
        view_479 = torch.ops.aten.view.default(bmm_104, [64, 6, 196, 9, 32]);  bmm_104 = None
        view_480 = torch.ops.aten.view.default(bmm_105, [64, 6, 196, 9, 9]);  bmm_105 = None
        alias_137 = torch.ops.aten.alias.default(alias_26);  alias_26 = None
        alias_138 = torch.ops.aten.alias.default(alias_137);  alias_137 = None
        mul_1028 = torch.ops.aten.mul.Tensor(view_480, alias_138);  view_480 = None
        sum_254 = torch.ops.aten.sum.dim_IntList(mul_1028, [-1], True)
        mul_1029 = torch.ops.aten.mul.Tensor(alias_138, sum_254);  alias_138 = sum_254 = None
        sub_235 = torch.ops.aten.sub.Tensor(mul_1028, mul_1029);  mul_1028 = mul_1029 = None
        mul_1030 = torch.ops.aten.mul.Tensor(sub_235, 0.1767766952966369);  sub_235 = None
        permute_635 = torch.ops.aten.permute.default(mul_1030, [0, 2, 1, 3, 4]);  mul_1030 = None
        clone_196 = torch.ops.aten.clone.default(permute_635, memory_format = torch.contiguous_format);  permute_635 = None
        _unsafe_view_240 = torch.ops.aten._unsafe_view.default(clone_196, [64, 14, 14, 486]);  clone_196 = None
        sum_255 = torch.ops.aten.sum.dim_IntList(_unsafe_view_240, [0, 1, 2], True)
        view_481 = torch.ops.aten.view.default(sum_255, [486]);  sum_255 = None
        view_482 = torch.ops.aten.view.default(_unsafe_view_240, [12544, 486]);  _unsafe_view_240 = None
        permute_636 = torch.ops.aten.permute.default(view_482, [1, 0])
        mm_243 = torch.ops.aten.mm.default(permute_636, view_10);  permute_636 = view_10 = None
        permute_637 = torch.ops.aten.permute.default(mm_243, [1, 0]);  mm_243 = None
        mm_244 = torch.ops.aten.mm.default(view_482, permute_638);  view_482 = permute_638 = None
        view_483 = torch.ops.aten.view.default(mm_244, [64, 14, 14, 192]);  mm_244 = None
        permute_639 = torch.ops.aten.permute.default(permute_637, [1, 0]);  permute_637 = None
        permute_640 = torch.ops.aten.permute.default(view_483, [0, 3, 1, 2]);  view_483 = None
        avg_pool2d_backward_2 = torch.ops.aten.avg_pool2d_backward.default(permute_640, permute_19, [2, 2], [2, 2], [0, 0], True, True, None);  permute_640 = permute_19 = None
        permute_641 = torch.ops.aten.permute.default(avg_pool2d_backward_2, [0, 2, 3, 1]);  avg_pool2d_backward_2 = None
        permute_642 = torch.ops.aten.permute.default(view_479, [0, 1, 4, 3, 2]);  view_479 = None
        clone_197 = torch.ops.aten.clone.default(permute_642, memory_format = torch.contiguous_format);  permute_642 = None
        _unsafe_view_241 = torch.ops.aten._unsafe_view.default(clone_197, [64, 1728, 196]);  clone_197 = None
        im2col_backward_2 = torch.ops.aten.im2col_backward.default(_unsafe_view_241, [28, 28], [3, 3], [1, 1], [1, 1], [2, 2]);  _unsafe_view_241 = None
        permute_643 = torch.ops.aten.permute.default(im2col_backward_2, [0, 2, 3, 1]);  im2col_backward_2 = None
        clone_198 = torch.ops.aten.clone.default(permute_643, memory_format = torch.contiguous_format);  permute_643 = None
        _unsafe_view_242 = torch.ops.aten._unsafe_view.default(clone_198, [50176, 192]);  clone_198 = None
        permute_644 = torch.ops.aten.permute.default(_unsafe_view_242, [1, 0])
        mm_245 = torch.ops.aten.mm.default(permute_644, view_8);  permute_644 = view_8 = None
        permute_645 = torch.ops.aten.permute.default(mm_245, [1, 0]);  mm_245 = None
        mm_246 = torch.ops.aten.mm.default(_unsafe_view_242, permute_646);  _unsafe_view_242 = permute_646 = None
        view_484 = torch.ops.aten.view.default(mm_246, [64, 28, 28, 192]);  mm_246 = None
        add_511 = torch.ops.aten.add.Tensor(permute_641, view_484);  permute_641 = view_484 = None
        permute_647 = torch.ops.aten.permute.default(permute_645, [1, 0]);  permute_645 = None
        mul_1032 = torch.ops.aten.mul.Tensor(add_511, primals_27);  primals_27 = None
        mul_1033 = torch.ops.aten.mul.Tensor(mul_1032, 192)
        sum_256 = torch.ops.aten.sum.dim_IntList(mul_1032, [3], True)
        mul_1034 = torch.ops.aten.mul.Tensor(mul_1032, mul_39);  mul_1032 = None
        sum_257 = torch.ops.aten.sum.dim_IntList(mul_1034, [3], True);  mul_1034 = None
        mul_1035 = torch.ops.aten.mul.Tensor(mul_39, sum_257);  sum_257 = None
        sub_237 = torch.ops.aten.sub.Tensor(mul_1033, sum_256);  mul_1033 = sum_256 = None
        sub_238 = torch.ops.aten.sub.Tensor(sub_237, mul_1035);  sub_237 = mul_1035 = None
        div_58 = torch.ops.aten.div.Tensor(reciprocal_6, 192);  reciprocal_6 = None
        mul_1036 = torch.ops.aten.mul.Tensor(div_58, sub_238);  div_58 = sub_238 = None
        mul_1037 = torch.ops.aten.mul.Tensor(add_511, mul_39);  mul_39 = None
        sum_258 = torch.ops.aten.sum.dim_IntList(mul_1037, [0, 1, 2]);  mul_1037 = None
        sum_259 = torch.ops.aten.sum.dim_IntList(add_511, [0, 1, 2]);  add_511 = None
        add_512 = torch.ops.aten.add.Tensor(add_510, mul_1036);  add_510 = mul_1036 = None
        sum_260 = torch.ops.aten.sum.dim_IntList(add_512, [0, 1, 2], True)
        view_485 = torch.ops.aten.view.default(sum_260, [192]);  sum_260 = None
        clone_200 = torch.ops.aten.clone.default(add_512, memory_format = torch.contiguous_format)
        _unsafe_view_243 = torch.ops.aten._unsafe_view.default(clone_200, [50176, 192]);  clone_200 = None
        permute_648 = torch.ops.aten.permute.default(_unsafe_view_243, [1, 0])
        mm_247 = torch.ops.aten.mm.default(permute_648, view_7);  permute_648 = view_7 = None
        permute_649 = torch.ops.aten.permute.default(mm_247, [1, 0]);  mm_247 = None
        mm_248 = torch.ops.aten.mm.default(_unsafe_view_243, permute_650);  _unsafe_view_243 = permute_650 = None
        view_486 = torch.ops.aten.view.default(mm_248, [64, 28, 28, 576]);  mm_248 = None
        permute_651 = torch.ops.aten.permute.default(permute_649, [1, 0]);  permute_649 = None
        mul_1054 = torch.ops.aten.mul.Tensor(view_486, add_519);  view_486 = add_519 = None
        sum_261 = torch.ops.aten.sum.dim_IntList(mul_1054, [0, 1, 2], True)
        view_487 = torch.ops.aten.view.default(sum_261, [576]);  sum_261 = None
        view_488 = torch.ops.aten.view.default(mul_1054, [50176, 576]);  mul_1054 = None
        permute_652 = torch.ops.aten.permute.default(view_488, [1, 0])
        mm_249 = torch.ops.aten.mm.default(permute_652, view_6);  permute_652 = view_6 = None
        permute_653 = torch.ops.aten.permute.default(mm_249, [1, 0]);  mm_249 = None
        mm_250 = torch.ops.aten.mm.default(view_488, permute_654);  view_488 = permute_654 = None
        view_489 = torch.ops.aten.view.default(mm_250, [64, 28, 28, 192]);  mm_250 = None
        permute_655 = torch.ops.aten.permute.default(permute_653, [1, 0]);  permute_653 = None
        mul_1056 = torch.ops.aten.mul.Tensor(view_489, primals_21);  primals_21 = None
        mul_1057 = torch.ops.aten.mul.Tensor(mul_1056, 192)
        sum_262 = torch.ops.aten.sum.dim_IntList(mul_1056, [3], True)
        mul_1058 = torch.ops.aten.mul.Tensor(mul_1056, mul_24);  mul_1056 = None
        sum_263 = torch.ops.aten.sum.dim_IntList(mul_1058, [3], True);  mul_1058 = None
        mul_1059 = torch.ops.aten.mul.Tensor(mul_24, sum_263);  sum_263 = None
        sub_241 = torch.ops.aten.sub.Tensor(mul_1057, sum_262);  mul_1057 = sum_262 = None
        sub_242 = torch.ops.aten.sub.Tensor(sub_241, mul_1059);  sub_241 = mul_1059 = None
        div_59 = torch.ops.aten.div.Tensor(reciprocal_4, 192);  reciprocal_4 = None
        mul_1060 = torch.ops.aten.mul.Tensor(div_59, sub_242);  div_59 = sub_242 = None
        mul_1061 = torch.ops.aten.mul.Tensor(view_489, mul_24);  mul_24 = None
        sum_264 = torch.ops.aten.sum.dim_IntList(mul_1061, [0, 1, 2]);  mul_1061 = None
        sum_265 = torch.ops.aten.sum.dim_IntList(view_489, [0, 1, 2]);  view_489 = None
        add_520 = torch.ops.aten.add.Tensor(add_512, mul_1060);  add_512 = mul_1060 = None
        sum_266 = torch.ops.aten.sum.dim_IntList(add_520, [0, 1, 2], True)
        view_490 = torch.ops.aten.view.default(sum_266, [192]);  sum_266 = None
        clone_202 = torch.ops.aten.clone.default(add_520, memory_format = torch.contiguous_format)
        _unsafe_view_244 = torch.ops.aten._unsafe_view.default(clone_202, [50176, 192]);  clone_202 = None
        permute_656 = torch.ops.aten.permute.default(_unsafe_view_244, [1, 0])
        mm_251 = torch.ops.aten.mm.default(permute_656, _unsafe_view_6);  permute_656 = _unsafe_view_6 = None
        permute_657 = torch.ops.aten.permute.default(mm_251, [1, 0]);  mm_251 = None
        mm_252 = torch.ops.aten.mm.default(_unsafe_view_244, permute_658);  _unsafe_view_244 = permute_658 = None
        view_491 = torch.ops.aten.view.default(mm_252, [64, 28, 28, 192]);  mm_252 = None
        permute_659 = torch.ops.aten.permute.default(permute_657, [1, 0]);  permute_657 = None
        permute_660 = torch.ops.aten.permute.default(view_491, [0, 3, 1, 2]);  view_491 = None
        col2im_backward_3 = torch.ops.aten.col2im_backward.default(permute_660, [3, 3], [1, 1], [1, 1], [2, 2]);  permute_660 = None
        view_492 = torch.ops.aten.view.default(col2im_backward_3, [64, 6, 32, 9, 196]);  col2im_backward_3 = None
        permute_661 = torch.ops.aten.permute.default(view_492, [0, 1, 4, 3, 2]);  view_492 = None
        clone_203 = torch.ops.aten.clone.default(permute_661, memory_format = torch.contiguous_format);  permute_661 = None
        _unsafe_view_245 = torch.ops.aten._unsafe_view.default(clone_203, [75264, 9, 32]);  clone_203 = None
        permute_662 = torch.ops.aten.permute.default(view_4, [0, 2, 1]);  view_4 = None
        bmm_106 = torch.ops.aten.bmm.default(permute_662, _unsafe_view_245);  permute_662 = None
        bmm_107 = torch.ops.aten.bmm.default(_unsafe_view_245, permute_663);  _unsafe_view_245 = permute_663 = None
        view_493 = torch.ops.aten.view.default(bmm_106, [64, 6, 196, 9, 32]);  bmm_106 = None
        view_494 = torch.ops.aten.view.default(bmm_107, [64, 6, 196, 9, 9]);  bmm_107 = None
        alias_139 = torch.ops.aten.alias.default(alias_15);  alias_15 = None
        alias_140 = torch.ops.aten.alias.default(alias_139);  alias_139 = None
        mul_1062 = torch.ops.aten.mul.Tensor(view_494, alias_140);  view_494 = None
        sum_267 = torch.ops.aten.sum.dim_IntList(mul_1062, [-1], True)
        mul_1063 = torch.ops.aten.mul.Tensor(alias_140, sum_267);  alias_140 = sum_267 = None
        sub_243 = torch.ops.aten.sub.Tensor(mul_1062, mul_1063);  mul_1062 = mul_1063 = None
        mul_1064 = torch.ops.aten.mul.Tensor(sub_243, 0.1767766952966369);  sub_243 = None
        permute_664 = torch.ops.aten.permute.default(mul_1064, [0, 2, 1, 3, 4]);  mul_1064 = None
        clone_204 = torch.ops.aten.clone.default(permute_664, memory_format = torch.contiguous_format);  permute_664 = None
        _unsafe_view_246 = torch.ops.aten._unsafe_view.default(clone_204, [64, 14, 14, 486]);  clone_204 = None
        sum_268 = torch.ops.aten.sum.dim_IntList(_unsafe_view_246, [0, 1, 2], True)
        view_495 = torch.ops.aten.view.default(sum_268, [486]);  sum_268 = None
        view_496 = torch.ops.aten.view.default(_unsafe_view_246, [12544, 486]);  _unsafe_view_246 = None
        permute_665 = torch.ops.aten.permute.default(view_496, [1, 0])
        mm_253 = torch.ops.aten.mm.default(permute_665, view_2);  permute_665 = view_2 = None
        permute_666 = torch.ops.aten.permute.default(mm_253, [1, 0]);  mm_253 = None
        mm_254 = torch.ops.aten.mm.default(view_496, permute_667);  view_496 = permute_667 = None
        view_497 = torch.ops.aten.view.default(mm_254, [64, 14, 14, 192]);  mm_254 = None
        permute_668 = torch.ops.aten.permute.default(permute_666, [1, 0]);  permute_666 = None
        permute_669 = torch.ops.aten.permute.default(view_497, [0, 3, 1, 2]);  view_497 = None
        avg_pool2d_backward_3 = torch.ops.aten.avg_pool2d_backward.default(permute_669, permute_5, [2, 2], [2, 2], [0, 0], True, True, None);  permute_669 = permute_5 = None
        permute_670 = torch.ops.aten.permute.default(avg_pool2d_backward_3, [0, 2, 3, 1]);  avg_pool2d_backward_3 = None
        permute_671 = torch.ops.aten.permute.default(view_493, [0, 1, 4, 3, 2]);  view_493 = None
        clone_205 = torch.ops.aten.clone.default(permute_671, memory_format = torch.contiguous_format);  permute_671 = None
        _unsafe_view_247 = torch.ops.aten._unsafe_view.default(clone_205, [64, 1728, 196]);  clone_205 = None
        im2col_backward_3 = torch.ops.aten.im2col_backward.default(_unsafe_view_247, [28, 28], [3, 3], [1, 1], [1, 1], [2, 2]);  _unsafe_view_247 = None
        permute_672 = torch.ops.aten.permute.default(im2col_backward_3, [0, 2, 3, 1]);  im2col_backward_3 = None
        clone_206 = torch.ops.aten.clone.default(permute_672, memory_format = torch.contiguous_format);  permute_672 = None
        _unsafe_view_248 = torch.ops.aten._unsafe_view.default(clone_206, [50176, 192]);  clone_206 = None
        permute_673 = torch.ops.aten.permute.default(_unsafe_view_248, [1, 0])
        mm_255 = torch.ops.aten.mm.default(permute_673, view);  permute_673 = view = None
        permute_674 = torch.ops.aten.permute.default(mm_255, [1, 0]);  mm_255 = None
        mm_256 = torch.ops.aten.mm.default(_unsafe_view_248, permute_675);  _unsafe_view_248 = permute_675 = None
        view_498 = torch.ops.aten.view.default(mm_256, [64, 28, 28, 192]);  mm_256 = None
        add_521 = torch.ops.aten.add.Tensor(permute_670, view_498);  permute_670 = view_498 = None
        permute_676 = torch.ops.aten.permute.default(permute_674, [1, 0]);  permute_674 = None
        mul_1066 = torch.ops.aten.mul.Tensor(add_521, primals_14);  primals_14 = None
        mul_1067 = torch.ops.aten.mul.Tensor(mul_1066, 192)
        sum_269 = torch.ops.aten.sum.dim_IntList(mul_1066, [3], True)
        mul_1068 = torch.ops.aten.mul.Tensor(mul_1066, mul_21);  mul_1066 = None
        sum_270 = torch.ops.aten.sum.dim_IntList(mul_1068, [3], True);  mul_1068 = None
        mul_1069 = torch.ops.aten.mul.Tensor(mul_21, sum_270);  sum_270 = None
        sub_245 = torch.ops.aten.sub.Tensor(mul_1067, sum_269);  mul_1067 = sum_269 = None
        sub_246 = torch.ops.aten.sub.Tensor(sub_245, mul_1069);  sub_245 = mul_1069 = None
        div_60 = torch.ops.aten.div.Tensor(reciprocal_3, 192);  reciprocal_3 = None
        mul_1070 = torch.ops.aten.mul.Tensor(div_60, sub_246);  div_60 = sub_246 = None
        mul_1071 = torch.ops.aten.mul.Tensor(add_521, mul_21);  mul_21 = None
        sum_271 = torch.ops.aten.sum.dim_IntList(mul_1071, [0, 1, 2]);  mul_1071 = None
        sum_272 = torch.ops.aten.sum.dim_IntList(add_521, [0, 1, 2]);  add_521 = None
        add_522 = torch.ops.aten.add.Tensor(add_520, mul_1070);  add_520 = mul_1070 = None
        permute_677 = torch.ops.aten.permute.default(add_522, [0, 3, 1, 2]);  add_522 = None
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(permute_677, relu_2, primals_12, [192], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  permute_677 = primals_12 = None
        getitem_133 = convolution_backward_1[0]
        getitem_134 = convolution_backward_1[1]
        getitem_135 = convolution_backward_1[2];  convolution_backward_1 = None
        alias_143 = torch.ops.aten.alias.default(relu_2);  relu_2 = None
        alias_144 = torch.ops.aten.alias.default(alias_143);  alias_143 = None
        alias_145 = torch.ops.aten.alias.default(alias_144);  alias_144 = None
        alias_146 = torch.ops.aten.alias.default(alias_145);  alias_145 = None
        le = torch.ops.aten.le.Scalar(alias_146, 0);  alias_146 = None
        scalar_tensor = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where = torch.ops.aten.where.self(le, scalar_tensor, getitem_133);  le = getitem_133 = None
        sum_273 = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        sub_247 = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_64);  convolution_2 = unsqueeze_64 = None
        mul_1072 = torch.ops.aten.mul.Tensor(where, sub_247)
        sum_274 = torch.ops.aten.sum.dim_IntList(mul_1072, [0, 2, 3]);  mul_1072 = None
        mul_1073 = torch.ops.aten.mul.Tensor(sum_273, 1.2456154336734693e-06)
        unsqueeze_65 = torch.ops.aten.unsqueeze.default(mul_1073, 0);  mul_1073 = None
        unsqueeze_66 = torch.ops.aten.unsqueeze.default(unsqueeze_65, 2);  unsqueeze_65 = None
        unsqueeze_67 = torch.ops.aten.unsqueeze.default(unsqueeze_66, 3);  unsqueeze_66 = None
        mul_1074 = torch.ops.aten.mul.Tensor(sum_274, 1.2456154336734693e-06)
        mul_1075 = torch.ops.aten.mul.Tensor(squeeze_17, squeeze_17)
        mul_1076 = torch.ops.aten.mul.Tensor(mul_1074, mul_1075);  mul_1074 = mul_1075 = None
        unsqueeze_68 = torch.ops.aten.unsqueeze.default(mul_1076, 0);  mul_1076 = None
        unsqueeze_69 = torch.ops.aten.unsqueeze.default(unsqueeze_68, 2);  unsqueeze_68 = None
        unsqueeze_70 = torch.ops.aten.unsqueeze.default(unsqueeze_69, 3);  unsqueeze_69 = None
        mul_1077 = torch.ops.aten.mul.Tensor(squeeze_17, primals_10);  primals_10 = None
        unsqueeze_71 = torch.ops.aten.unsqueeze.default(mul_1077, 0);  mul_1077 = None
        unsqueeze_72 = torch.ops.aten.unsqueeze.default(unsqueeze_71, 2);  unsqueeze_71 = None
        unsqueeze_73 = torch.ops.aten.unsqueeze.default(unsqueeze_72, 3);  unsqueeze_72 = None
        mul_1078 = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_70);  sub_247 = unsqueeze_70 = None
        sub_249 = torch.ops.aten.sub.Tensor(where, mul_1078);  where = mul_1078 = None
        sub_250 = torch.ops.aten.sub.Tensor(sub_249, unsqueeze_67);  sub_249 = unsqueeze_67 = None
        mul_1079 = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_73);  sub_250 = unsqueeze_73 = None
        mul_1080 = torch.ops.aten.mul.Tensor(sum_274, squeeze_17);  sum_274 = squeeze_17 = None
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_1079, relu_1, primals_9, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1079 = primals_9 = None
        getitem_136 = convolution_backward_2[0]
        getitem_137 = convolution_backward_2[1];  convolution_backward_2 = None
        alias_149 = torch.ops.aten.alias.default(relu_1);  relu_1 = None
        alias_150 = torch.ops.aten.alias.default(alias_149);  alias_149 = None
        alias_151 = torch.ops.aten.alias.default(alias_150);  alias_150 = None
        alias_152 = torch.ops.aten.alias.default(alias_151);  alias_151 = None
        le_1 = torch.ops.aten.le.Scalar(alias_152, 0);  alias_152 = None
        where_1 = torch.ops.aten.where.self(le_1, scalar_tensor, getitem_136);  le_1 = getitem_136 = None
        sum_275 = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
        sub_251 = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_76);  convolution_1 = unsqueeze_76 = None
        mul_1081 = torch.ops.aten.mul.Tensor(where_1, sub_251)
        sum_276 = torch.ops.aten.sum.dim_IntList(mul_1081, [0, 2, 3]);  mul_1081 = None
        mul_1082 = torch.ops.aten.mul.Tensor(sum_275, 1.2456154336734693e-06)
        unsqueeze_77 = torch.ops.aten.unsqueeze.default(mul_1082, 0);  mul_1082 = None
        unsqueeze_78 = torch.ops.aten.unsqueeze.default(unsqueeze_77, 2);  unsqueeze_77 = None
        unsqueeze_79 = torch.ops.aten.unsqueeze.default(unsqueeze_78, 3);  unsqueeze_78 = None
        mul_1083 = torch.ops.aten.mul.Tensor(sum_276, 1.2456154336734693e-06)
        mul_1084 = torch.ops.aten.mul.Tensor(squeeze_11, squeeze_11)
        mul_1085 = torch.ops.aten.mul.Tensor(mul_1083, mul_1084);  mul_1083 = mul_1084 = None
        unsqueeze_80 = torch.ops.aten.unsqueeze.default(mul_1085, 0);  mul_1085 = None
        unsqueeze_81 = torch.ops.aten.unsqueeze.default(unsqueeze_80, 2);  unsqueeze_80 = None
        unsqueeze_82 = torch.ops.aten.unsqueeze.default(unsqueeze_81, 3);  unsqueeze_81 = None
        mul_1086 = torch.ops.aten.mul.Tensor(squeeze_11, primals_7);  primals_7 = None
        unsqueeze_83 = torch.ops.aten.unsqueeze.default(mul_1086, 0);  mul_1086 = None
        unsqueeze_84 = torch.ops.aten.unsqueeze.default(unsqueeze_83, 2);  unsqueeze_83 = None
        unsqueeze_85 = torch.ops.aten.unsqueeze.default(unsqueeze_84, 3);  unsqueeze_84 = None
        mul_1087 = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_82);  sub_251 = unsqueeze_82 = None
        sub_253 = torch.ops.aten.sub.Tensor(where_1, mul_1087);  where_1 = mul_1087 = None
        sub_254 = torch.ops.aten.sub.Tensor(sub_253, unsqueeze_79);  sub_253 = unsqueeze_79 = None
        mul_1088 = torch.ops.aten.mul.Tensor(sub_254, unsqueeze_85);  sub_254 = unsqueeze_85 = None
        mul_1089 = torch.ops.aten.mul.Tensor(sum_276, squeeze_11);  sum_276 = squeeze_11 = None
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_1088, relu, primals_6, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1088 = primals_6 = None
        getitem_139 = convolution_backward_3[0]
        getitem_140 = convolution_backward_3[1];  convolution_backward_3 = None
        alias_155 = torch.ops.aten.alias.default(relu);  relu = None
        alias_156 = torch.ops.aten.alias.default(alias_155);  alias_155 = None
        alias_157 = torch.ops.aten.alias.default(alias_156);  alias_156 = None
        alias_158 = torch.ops.aten.alias.default(alias_157);  alias_157 = None
        le_2 = torch.ops.aten.le.Scalar(alias_158, 0);  alias_158 = None
        where_2 = torch.ops.aten.where.self(le_2, scalar_tensor, getitem_139);  le_2 = scalar_tensor = getitem_139 = None
        sum_277 = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
        sub_255 = torch.ops.aten.sub.Tensor(convolution, unsqueeze_88);  convolution = unsqueeze_88 = None
        mul_1090 = torch.ops.aten.mul.Tensor(where_2, sub_255)
        sum_278 = torch.ops.aten.sum.dim_IntList(mul_1090, [0, 2, 3]);  mul_1090 = None
        mul_1091 = torch.ops.aten.mul.Tensor(sum_277, 1.2456154336734693e-06)
        unsqueeze_89 = torch.ops.aten.unsqueeze.default(mul_1091, 0);  mul_1091 = None
        unsqueeze_90 = torch.ops.aten.unsqueeze.default(unsqueeze_89, 2);  unsqueeze_89 = None
        unsqueeze_91 = torch.ops.aten.unsqueeze.default(unsqueeze_90, 3);  unsqueeze_90 = None
        mul_1092 = torch.ops.aten.mul.Tensor(sum_278, 1.2456154336734693e-06)
        mul_1093 = torch.ops.aten.mul.Tensor(squeeze_5, squeeze_5)
        mul_1094 = torch.ops.aten.mul.Tensor(mul_1092, mul_1093);  mul_1092 = mul_1093 = None
        unsqueeze_92 = torch.ops.aten.unsqueeze.default(mul_1094, 0);  mul_1094 = None
        unsqueeze_93 = torch.ops.aten.unsqueeze.default(unsqueeze_92, 2);  unsqueeze_92 = None
        unsqueeze_94 = torch.ops.aten.unsqueeze.default(unsqueeze_93, 3);  unsqueeze_93 = None
        mul_1095 = torch.ops.aten.mul.Tensor(squeeze_5, primals_4);  primals_4 = None
        unsqueeze_95 = torch.ops.aten.unsqueeze.default(mul_1095, 0);  mul_1095 = None
        unsqueeze_96 = torch.ops.aten.unsqueeze.default(unsqueeze_95, 2);  unsqueeze_95 = None
        unsqueeze_97 = torch.ops.aten.unsqueeze.default(unsqueeze_96, 3);  unsqueeze_96 = None
        mul_1096 = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_94);  sub_255 = unsqueeze_94 = None
        sub_257 = torch.ops.aten.sub.Tensor(where_2, mul_1096);  where_2 = mul_1096 = None
        sub_258 = torch.ops.aten.sub.Tensor(sub_257, unsqueeze_91);  sub_257 = unsqueeze_91 = None
        mul_1097 = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_97);  sub_258 = unsqueeze_97 = None
        mul_1098 = torch.ops.aten.mul.Tensor(sum_278, squeeze_5);  sum_278 = squeeze_5 = None
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_1097, primals_261, primals_3, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1097 = primals_261 = primals_3 = None
        getitem_143 = convolution_backward_4[1];  convolution_backward_4 = None
        return [sum_220, sum_51, getitem_143, mul_1098, sum_277, getitem_140, mul_1089, sum_275, getitem_137, mul_1080, sum_273, getitem_134, getitem_135, sum_271, sum_272, permute_676, permute_668, view_495, permute_659, view_490, sum_264, sum_265, permute_655, view_487, permute_651, view_485, sum_258, sum_259, permute_647, permute_639, view_481, permute_630, view_476, sum_251, sum_252, permute_626, view_473, permute_622, view_471, sum_245, sum_246, permute_618, permute_610, view_467, permute_601, view_462, sum_238, sum_239, permute_597, view_459, permute_593, view_457, sum_232, sum_233, permute_589, permute_581, view_453, permute_572, view_448, sum_225, sum_226, permute_568, view_445, permute_564, view_443, getitem_131, getitem_132, sum_218, sum_219, permute_558, permute_547, view_431, sum_212, sum_213, permute_543, view_428, permute_539, view_425, sum_206, sum_207, permute_535, permute_524, view_413, sum_200, sum_201, permute_520, view_410, permute_516, view_407, sum_194, sum_195, permute_512, permute_501, view_395, sum_188, sum_189, permute_497, view_392, permute_493, view_389, sum_182, sum_183, permute_489, permute_478, view_377, sum_176, sum_177, permute_474, view_374, permute_470, view_371, sum_170, sum_171, permute_466, permute_455, view_359, sum_164, sum_165, permute_451, view_356, permute_447, view_353, sum_158, sum_159, permute_443, permute_432, view_341, sum_152, sum_153, permute_428, view_338, permute_424, view_335, sum_146, sum_147, permute_420, permute_409, view_323, sum_140, sum_141, permute_405, view_320, permute_401, view_317, sum_134, sum_135, permute_397, permute_386, view_305, sum_128, sum_129, permute_382, view_302, permute_378, view_299, sum_122, sum_123, permute_374, permute_363, view_287, sum_116, sum_117, permute_359, view_284, permute_355, view_281, sum_110, sum_111, permute_351, permute_340, view_269, sum_104, sum_105, permute_336, view_266, permute_332, view_263, sum_98, sum_99, permute_328, permute_317, view_251, sum_92, sum_93, permute_313, view_248, permute_309, view_245, sum_86, sum_87, permute_305, permute_294, view_233, sum_80, sum_81, permute_290, view_230, permute_286, view_227, sum_74, sum_75, permute_282, permute_271, view_215, sum_68, sum_69, permute_267, view_212, permute_263, view_209, sum_62, sum_63, permute_259, permute_248, view_197, sum_56, sum_57, permute_244, view_194, permute_240, view_192, sum_49, sum_50, permute_236, permute_231, permute_221, view_176, sum_43, sum_44, permute_217, view_173, permute_213, view_170, sum_37, sum_38, permute_209, permute_204, permute_194, view_154, sum_31, sum_32, permute_190, view_151, permute_186, view_148, sum_25, sum_26, permute_182, view_146, permute_178, view_143, None, None, None, None, None, None, None, None, None, None]


peak_mem = lambda :  torch.cuda.max_memory_allocated() / 10**9
print_peak_mem = lambda x: print(f"{x}: {peak_mem():.2f} GB")

torch.cuda.reset_peak_memory_stats()
print_peak_mem("Start of file")

args = [((64, 3, 7, 7), (147, 49, 7, 1), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((64, 64, 3, 3), (576, 9, 3, 1), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((64, 64, 3, 3), (576, 9, 3, 1), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((192, 64, 4, 4), (1024, 16, 4, 1), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((192,), (1,), torch.float32, 'cuda'), ((384, 192, 2, 2), (768, 4, 2, 1), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((384,), (1,), torch.float32, 'cuda'), ((64, 3, 224, 224), (150528, 50176, 224, 1), torch.float32, 'cuda'), ((64, 64, 112, 112), (802816, 12544, 112, 1), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((64, 64, 112, 112), (802816, 12544, 112, 1), torch.float32, 'cuda'), ((64, 64, 112, 112), (802816, 12544, 112, 1), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((64, 64, 112, 112), (802816, 12544, 112, 1), torch.float32, 'cuda'), ((64, 64, 112, 112), (802816, 12544, 112, 1), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((64, 64, 112, 112), (802816, 12544, 112, 1), torch.float32, 'cuda'), ((64, 28, 28, 192), (150528, 5376, 192, 1), torch.float32, 'cuda'), ((64, 28, 28, 1), (784, 28, 1, 1), torch.float32, 'cuda'), ((64, 28, 28, 1), (784, 28, 1, 1), torch.float32, 'cuda'), ((12544, 192), (192, 1), torch.float32, 'cuda'), ((64, 6, 196, 9, 9), (95256, 15876, 81, 9, 1), torch.float32, 'cuda'), ((50176, 192), (192, 1), torch.float32, 'cuda'), ((64, 28, 28, 192), (150528, 5376, 192, 1), torch.float32, 'cuda'), ((64, 28, 28, 1), (784, 28, 1, 1), torch.float32, 'cuda'), ((64, 28, 28, 1), (784, 28, 1, 1), torch.float32, 'cuda'), ((50176, 576), (576, 1), torch.float32, 'cuda'), ((64, 28, 28, 192), (150528, 5376, 192, 1), torch.float32, 'cuda'), ((64, 28, 28, 1), (784, 28, 1, 1), torch.float32, 'cuda'), ((64, 28, 28, 1), (784, 28, 1, 1), torch.float32, 'cuda'), ((12544, 192), (192, 1), torch.float32, 'cuda'), ((64, 6, 196, 9, 9), (95256, 15876, 81, 9, 1), torch.float32, 'cuda'), ((50176, 192), (192, 1), torch.float32, 'cuda'), ((64, 28, 28, 192), (150528, 5376, 192, 1), torch.float32, 'cuda'), ((64, 28, 28, 1), (784, 28, 1, 1), torch.float32, 'cuda'), ((64, 28, 28, 1), (784, 28, 1, 1), torch.float32, 'cuda'), ((50176, 576), (576, 1), torch.float32, 'cuda'), ((64, 28, 28, 192), (150528, 5376, 192, 1), torch.float32, 'cuda'), ((64, 28, 28, 1), (784, 28, 1, 1), torch.float32, 'cuda'), ((64, 28, 28, 1), (784, 28, 1, 1), torch.float32, 'cuda'), ((12544, 192), (192, 1), torch.float32, 'cuda'), ((64, 6, 196, 9, 9), (95256, 15876, 81, 9, 1), torch.float32, 'cuda'), ((50176, 192), (192, 1), torch.float32, 'cuda'), ((64, 28, 28, 192), (150528, 5376, 192, 1), torch.float32, 'cuda'), ((64, 28, 28, 1), (784, 28, 1, 1), torch.float32, 'cuda'), ((64, 28, 28, 1), (784, 28, 1, 1), torch.float32, 'cuda'), ((50176, 576), (576, 1), torch.float32, 'cuda'), ((64, 28, 28, 192), (150528, 5376, 192, 1), torch.float32, 'cuda'), ((64, 28, 28, 1), (784, 28, 1, 1), torch.float32, 'cuda'), ((64, 28, 28, 1), (784, 28, 1, 1), torch.float32, 'cuda'), ((12544, 192), (192, 1), torch.float32, 'cuda'), ((64, 6, 196, 9, 9), (95256, 15876, 81, 9, 1), torch.float32, 'cuda'), ((50176, 192), (192, 1), torch.float32, 'cuda'), ((64, 28, 28, 192), (150528, 5376, 192, 1), torch.float32, 'cuda'), ((64, 28, 28, 1), (784, 28, 1, 1), torch.float32, 'cuda'), ((64, 28, 28, 1), (784, 28, 1, 1), torch.float32, 'cuda'), ((50176, 576), (576, 1), torch.float32, 'cuda'), ((64, 192, 28, 28), (150528, 784, 28, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 12, 196, 196), (460992, 38416, 196, 1), torch.float32, 'cuda'), ((12544, 384), (384, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((12544, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 12, 196, 196), (460992, 38416, 196, 1), torch.float32, 'cuda'), ((12544, 384), (384, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((12544, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 12, 196, 196), (460992, 38416, 196, 1), torch.float32, 'cuda'), ((12544, 384), (384, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((12544, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 12, 196, 196), (460992, 38416, 196, 1), torch.float32, 'cuda'), ((12544, 384), (384, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((12544, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 12, 196, 196), (460992, 38416, 196, 1), torch.float32, 'cuda'), ((12544, 384), (384, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((12544, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 12, 196, 196), (460992, 38416, 196, 1), torch.float32, 'cuda'), ((12544, 384), (384, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((12544, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 12, 196, 196), (460992, 38416, 196, 1), torch.float32, 'cuda'), ((12544, 384), (384, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((12544, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 12, 196, 196), (460992, 38416, 196, 1), torch.float32, 'cuda'), ((12544, 384), (384, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((12544, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 12, 196, 196), (460992, 38416, 196, 1), torch.float32, 'cuda'), ((12544, 384), (384, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((12544, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 12, 196, 196), (460992, 38416, 196, 1), torch.float32, 'cuda'), ((12544, 384), (384, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((12544, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 12, 196, 196), (460992, 38416, 196, 1), torch.float32, 'cuda'), ((12544, 384), (384, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((12544, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 12, 196, 196), (460992, 38416, 196, 1), torch.float32, 'cuda'), ((12544, 384), (384, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((12544, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 12, 196, 196), (460992, 38416, 196, 1), torch.float32, 'cuda'), ((12544, 384), (384, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((12544, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 12, 196, 196), (460992, 38416, 196, 1), torch.float32, 'cuda'), ((12544, 384), (384, 1), torch.float32, 'cuda'), ((64, 14, 14, 384), (75264, 5376, 384, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((64, 14, 14, 1), (196, 14, 1, 1), torch.float32, 'cuda'), ((12544, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 197, 384), (75648, 384, 1), torch.float32, 'cuda'), ((64, 197, 1), (197, 1, 1), torch.float32, 'cuda'), ((64, 197, 1), (197, 1, 1), torch.float32, 'cuda'), ((64, 384), (75648, 1), torch.float32, 'cuda'), ((768, 1, 197), (197, 197, 1), torch.float32, 'cuda'), ((64, 12, 1, 1), (12, 1, 1, 1), torch.float32, 'cuda'), ((64, 12, 1, 1), (12, 1, 1, 1), torch.float32, 'cuda'), ((64, 384), (384, 1), torch.float32, 'cuda'), ((64, 1, 384), (384, 384, 1), torch.float32, 'cuda'), ((64, 1, 1), (1, 1, 1), torch.float32, 'cuda'), ((64, 1, 1), (1, 1, 1), torch.float32, 'cuda'), ((64, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 197, 384), (75648, 384, 1), torch.float32, 'cuda'), ((64, 197, 1), (197, 1, 1), torch.float32, 'cuda'), ((64, 197, 1), (197, 1, 1), torch.float32, 'cuda'), ((64, 384), (75648, 1), torch.float32, 'cuda'), ((768, 1, 197), (197, 197, 1), torch.float32, 'cuda'), ((64, 12, 1, 1), (12, 1, 1, 1), torch.float32, 'cuda'), ((64, 12, 1, 1), (12, 1, 1, 1), torch.float32, 'cuda'), ((64, 384), (384, 1), torch.float32, 'cuda'), ((64, 1, 384), (384, 384, 1), torch.float32, 'cuda'), ((64, 1, 1), (1, 1, 1), torch.float32, 'cuda'), ((64, 1, 1), (1, 1, 1), torch.float32, 'cuda'), ((64, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 197, 384), (75648, 384, 1), torch.float32, 'cuda'), ((64, 197, 1), (197, 1, 1), torch.float32, 'cuda'), ((64, 197, 1), (197, 1, 1), torch.float32, 'cuda'), ((64, 384), (75648, 1), torch.float32, 'cuda'), ((12544, 384), (384, 1), torch.float32, 'cuda'), ((64, 1, 1000), (1000, 1000, 1), torch.int64, 'cuda'), ((1000, 384), (384, 1), torch.float32, 'cuda'), ((1000, 384), (384, 1), torch.float32, 'cuda'), ((384, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 1, 1152), (1152, 1152, 1), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 384), (384, 1), torch.float32, 'cuda'), ((768, 32, 197), (6304, 1, 32), torch.float32, 'cuda'), ((768, 32, 1), (32, 1, 32), torch.float32, 'cuda'), ((768, 197, 32), (6304, 1, 197), torch.float32, 'cuda'), ((384, 384), (384, 1), torch.float32, 'cuda'), ((768, 384), (384, 1), torch.float32, 'cuda'), ((384, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 1, 1152), (1152, 1152, 1), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 384), (384, 1), torch.float32, 'cuda'), ((768, 32, 197), (6304, 1, 32), torch.float32, 'cuda'), ((768, 32, 1), (32, 1, 32), torch.float32, 'cuda'), ((768, 197, 32), (6304, 1, 197), torch.float32, 'cuda'), ((384, 384), (384, 1), torch.float32, 'cuda'), ((768, 384), (384, 1), torch.float32, 'cuda'), ((384, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 1152), (225792, 16128, 1152, 1), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 384), (384, 1), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 196, 32), (6272, 1, 196), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 1152), (225792, 16128, 1152, 1), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 384), (384, 1), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 196, 32), (6272, 1, 196), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 1152), (225792, 16128, 1152, 1), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 384), (384, 1), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 196, 32), (6272, 1, 196), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 1152), (225792, 16128, 1152, 1), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 384), (384, 1), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 196, 32), (6272, 1, 196), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 1152), (225792, 16128, 1152, 1), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 384), (384, 1), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 196, 32), (6272, 1, 196), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 1152), (225792, 16128, 1152, 1), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 384), (384, 1), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 196, 32), (6272, 1, 196), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 1152), (225792, 16128, 1152, 1), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 384), (384, 1), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 196, 32), (6272, 1, 196), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 1152), (225792, 16128, 1152, 1), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 384), (384, 1), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 196, 32), (6272, 1, 196), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 1152), (225792, 16128, 1152, 1), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 384), (384, 1), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 196, 32), (6272, 1, 196), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 1152), (225792, 16128, 1152, 1), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 384), (384, 1), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 196, 32), (6272, 1, 196), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 1152), (225792, 16128, 1152, 1), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 384), (384, 1), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 196, 32), (6272, 1, 196), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 1152), (225792, 16128, 1152, 1), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 384), (384, 1), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 196, 32), (6272, 1, 196), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 1152), (225792, 16128, 1152, 1), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 384), (384, 1), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 196, 32), (6272, 1, 196), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 1152), (1152, 1), torch.float32, 'cuda'), ((64, 14, 14, 1152), (225792, 16128, 1152, 1), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((384, 384), (384, 1), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 32, 196), (6272, 1, 32), torch.float32, 'cuda'), ((768, 196, 32), (6272, 1, 196), torch.float32, 'cuda'), ((1152, 384), (384, 1), torch.float32, 'cuda'), ((192, 576), (576, 1), torch.float32, 'cuda'), ((64, 28, 28, 576), (451584, 16128, 576, 1), torch.float32, 'cuda'), ((576, 192), (192, 1), torch.float32, 'cuda'), ((192, 192), (192, 1), torch.float32, 'cuda'), ((75264, 32, 9), (288, 1, 32), torch.float32, 'cuda'), ((486, 192), (192, 1), torch.float32, 'cuda'), ((192, 192), (192, 1), torch.float32, 'cuda'), ((192, 576), (576, 1), torch.float32, 'cuda'), ((64, 28, 28, 576), (451584, 16128, 576, 1), torch.float32, 'cuda'), ((576, 192), (192, 1), torch.float32, 'cuda'), ((192, 192), (192, 1), torch.float32, 'cuda'), ((75264, 32, 9), (288, 1, 32), torch.float32, 'cuda'), ((486, 192), (192, 1), torch.float32, 'cuda'), ((192, 192), (192, 1), torch.float32, 'cuda'), ((192, 576), (576, 1), torch.float32, 'cuda'), ((64, 28, 28, 576), (451584, 16128, 576, 1), torch.float32, 'cuda'), ((576, 192), (192, 1), torch.float32, 'cuda'), ((192, 192), (192, 1), torch.float32, 'cuda'), ((75264, 32, 9), (288, 1, 32), torch.float32, 'cuda'), ((486, 192), (192, 1), torch.float32, 'cuda'), ((192, 192), (192, 1), torch.float32, 'cuda'), ((192, 576), (576, 1), torch.float32, 'cuda'), ((64, 28, 28, 576), (451584, 16128, 576, 1), torch.float32, 'cuda'), ((576, 192), (192, 1), torch.float32, 'cuda'), ((192, 192), (192, 1), torch.float32, 'cuda'), ((75264, 32, 9), (288, 1, 32), torch.float32, 'cuda'), ((486, 192), (192, 1), torch.float32, 'cuda'), ((192, 192), (192, 1), torch.float32, 'cuda'), ((1, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cuda'), ((1, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cuda'), ((1, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cuda'), ((64, 1000), (1000, 1), torch.float32, 'cuda')]
args = [rand_strided(shape, stride, dtype, device) for shape, stride, dtype, device in args]
print_peak_mem("Inputs setup")
torch.cuda.reset_peak_memory_stats()
mod = make_fx(Repro())(*args)
print_peak_mem("make_fx done")


from torchinductor.compile_fx import compile_fx_inner

torch.cuda.reset_peak_memory_stats()
print("\n\n------")
print_peak_mem("Before running eager")
start_mem = peak_mem()
mod(*args)
print_peak_mem("After running eager")
end_mem = peak_mem()
print(f"Increase in peak memory in eager: {end_mem - start_mem:.2f}")

torch.cuda.reset_peak_memory_stats()
print("\n\n------")
print_peak_mem("Before running inductor")
start_mem = peak_mem()
compiled = compile_fx_inner(mod, args)
print_peak_mem("inductor compiler wrapped")
compiled(*args)
print_peak_mem("after running inductor")
end_mem = peak_mem()
print(f"Increase in peak memory in inductor: {end_mem - start_mem:.2f}")




# from functools import partial
# from torchdynamo.debug_utils import (
#     isolate_fails,
#     dump_compiler_graph_state,
# )
# from functorch.compile import minifier

# env_variables = {"CUDA_VISIBLE_DEVICES": "0"}

# def has_high_peak_memory(gm, inputs):
#     torch.cuda.reset_peak_memory_stats()
#     print("\n\n------")
#     print_peak_mem("Before running eager")
#     start_mem = peak_mem()
#     gm(*inputs)
#     print_peak_mem("After running eager")
#     end_mem = peak_mem()
#     eager_mem = end_mem - start_mem
#     print(f"Increase in peak memory in eager: {end_mem - start_mem:.2f}")

#     torch.cuda.reset_peak_memory_stats()
#     print("\n\n------")
#     print_peak_mem("Before running inductor")
#     start_mem = peak_mem()
#     compiled = compile_fx_inner(gm, inputs)
#     print_peak_mem("inductor compiler wrapped")
#     compiled(*inputs)
#     print_peak_mem("after running inductor")
#     end_mem = peak_mem()
#     inductor_mem = end_mem - start_mem
#     print(f"Increase in peak memory in inductor: {end_mem - start_mem:.2f}")

#     if inductor_mem > 2 * eager_mem:
#         return True
#     return False





# minifier(
#     mod,
#     args,
#     module_fails=has_high_peak_memory,
#     dump_state=partial(dump_compiler_graph_state, compiler_name="inductor"),
# )