import logging
import warnings
from typing import Dict, Iterable, Union, Any

import onnx
import onnx.numpy_helper
from onnx.numpy_helper import to_array
import torch

import glob
import abc
import os

from pathlib import Path

__all__ = [
  Path(f).stem
  for f in glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))
  if os.path.isfile(f) and not f.endswith('__init__.py')
] + ["OpCodeGenerator", "CustomOpCodeGenerator"]


class OpCodeGenerator:
  __OP_DOMAIN__: str = ""
  __OP_VERSION__: int = 1

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    self.onnx_ver = onnx_ver
    self.torch_ver = torch_ver
    self.domain = self.__OP_DOMAIN__ or ""
    self.attr_default: Dict[str, Any] = {}
    self.op_ver = self.__OP_VERSION__ or 1

    # Should inherit from ModelCodeGenerator
    self.rename_helper = None
    self.tensor_inplace = None

    if self.domain == "":
      self.onnx_op = self.__class__.__name__.replace("OpCodeGenerator", "")
      self.schema = onnx.defs.get_schema(self.onnx_op,
                                         max_inclusive_version=onnx_ver)

      if self.schema is not None:
        self.op_ver = self.schema.since_version
        for a, i in self.schema.attributes.items():
          try:
            default_value = onnx.helper.get_attribute_value(i.default_value)
            self.attr_default[a] = default_value
          except Exception as e:
            logging.warning(
                f"Cannot get default value for {a} of {self.onnx_op}.")

  def gen(self, node, value_infos, initializers) -> Dict[str, Union[Iterable[str], str]]:
    """
    Generate code for a node.

    Parameters
    ----------
    node : onnx.NodeProto
    value_infos : Dict[str, onnx.ValueInfoProto]
    initializers : Dict[str, onnx.TensorProto]

    Returns
    -------
    code : Dict[str, Union[Iterable[str], str]]
      Contains `init` and `forward` code.
    """
    raise NotImplementedError("Base class of OpCodeGenerator should not be used.")

  def get_attr_value_dict(self, node):
    attr_value_dict = {}
    for a in node.attribute:
      attr_value_dict[a.name] = onnx.helper.get_attribute_value(a)
    attr_value_dict = dict(
        list(self.attr_default.items()) + list(attr_value_dict.items()))
    return attr_value_dict

  def gen_input_output_string(self,
                              node,
                              initializers,
                              rename_helper,
                              tensor_inplace=False,
                              input_num=None,
                              output_num=None):
    inputs_str, outputs_str = [], []
    input_num, output_num = input_num or len(node.input), output_num or len(
        node.output)
    for idx, (num, f, ls) in enumerate(
        ((input_num, node.input, inputs_str), (output_num, node.output,
                                               outputs_str))):
      for i in range(num):
        # tensor_inplace condition:
        # idx == 1: output
        # i == 0: first output tensor (Currently only support first tensor inplace)
        # node.input[0] not in initializers: Could not inplace initializer
        # rename_helper.tensor_name_counter[f[i]] == 2: output tensor 0 should only be counted twice
        # rename_helper.tensor_name_counter[node.input[0]] == 2: input tensor 0 should only be counted twice
        if idx == 1 \
            and i == 0 \
            and tensor_inplace \
            and len(node.input) > 0 \
            and node.input[0] not in initializers \
            and rename_helper.tensor_name_counter[f[i]] == 2 \
            and rename_helper.tensor_name_counter[node.input[0]] == 2:
          tensor_name = node.input[0]
          rename_helper.tensor_name_mapping[
              f[i]] = rename_helper.get_tensor_name(tensor_name)
        else:
          tensor_name = f[i]
        formatter = "{}"
        if tensor_name in initializers:
          formatter = "self._vars[\"{}\"]"
        s = formatter.format(rename_helper.get_tensor_name(tensor_name))
        ls.append(s)

    return inputs_str, outputs_str

  @staticmethod
  def gen_params_str(**kwargs):
    params = []
    for k, v in kwargs.items():
      v_str = v if type(v) == str else v.__repr__()
      params.append(f"'{k}': {v_str}")
    return ', '.join(params).__repr__()[1:-1]

  def check_in_init(self, targets, initializers):
    lacks = []
    rs = [None] * len(targets)
    for i, (t, n) in enumerate(targets):
      init = initializers.get(n, None)
      if init is None:
        lacks.append(n)
      rs[i] = init
    if lacks:
      raise Exception(
          f"Currently {self.__class__} only support all of {lacks.__repr__()} is in initializers."
      )
    return rs

  @staticmethod
  def get_shape(value_name: str, value_infos: Dict[str, onnx.ValueInfoProto]):
    if value_name not in value_infos:
      return None
    shape = []
    for d in value_infos[value_name].type.tensor_type.shape.dim:
      if d.dim_param != "":
        shape.append(-1)
      else:
        shape.append(d.dim_value)
    return shape


class ReduceOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(ReduceOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def _get_dim(self, attr_value_dict, d, node, initializers):
    if "axes" in attr_value_dict:
      dim = attr_value_dict["axes"]
    else:
      dim = list(range(d))
      if len(node.input) > 1:
        dim = initializers.get(node.input[1], None)
        assert dim is not None, "Currently ReduceOpCodeGenerator only support all of [axes] is in initializers."
        dim = list(to_array(dim))
    return dim


class CustomOpCodeGenerator(OpCodeGenerator):
  __OP_DOMAIN__ = "custom"

  def __init__(self, torch_ver=torch.__version__):
    super(CustomOpCodeGenerator, self).__init__(torch_ver=torch_ver)
    self.attr_default = self.gen_default_attr_values()

  @staticmethod
  @abc.abstractmethod
  def gen_default_attr_values() -> Dict[str, Any]:
    return dict()


# Deprecated: cache for existing opcode generator
__op_gen_dict = {}


def get_op_code_generator(op, **kwargs):
  """
  Get OpCodeGenerator by op name.

  If op is not in cache, create a new one. If op is in cache, return the cached one.

  Parameters
  ----------
  op : str
  **kwargs : dict, optional

  Returns
  -------
  _ : OpCodeGenerator or None
    If op is not supported or not found, return None.
  """
  warnings.warn("get_op_code_generator is deprecated, use code_generator_loader instead.", DeprecationWarning)
  op_code_gen_name = "{}OpCodeGenerator".format(op)
  if op_code_gen_name in __op_gen_dict:
    return __op_gen_dict[op_code_gen_name]
  mod = globals().get(op, None)
  if mod is None:
    return None
  __op_gen_dict[op_code_gen_name] = getattr(mod, op_code_gen_name)(**kwargs)
  return __op_gen_dict[op_code_gen_name]


def clear_op_code_generator():
  """
  Clear all cached opcode generators.
  """
  warnings.warn("clear_op_code_generator is deprecated, use code_generator_loader instead.", DeprecationWarning)
  global __op_gen_dict
  __op_gen_dict = {}
