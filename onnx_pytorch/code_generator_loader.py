import os
import inspect
import logging
import importlib.util
from types import ModuleType
from typing import List, Tuple, Dict, Any, Optional, NamedTuple, Union

import onnx_pytorch.op_code_generators
from onnx_pytorch.op_code_generators import *


class CodeGeneratorPack(NamedTuple):
  version: int
  module: ModuleType
  class_name: str

  def new(self) -> OpCodeGenerator:
    clazz = getattr(self.module, self.class_name)
    assert inspect.isclass(clazz), f"{self.class_name} is not a class"
    assert issubclass(clazz, OpCodeGenerator), f"{self.class_name} is not a subclass of OpCodeGenerator"
    return clazz()


OpSet = Dict[str, List[CodeGeneratorPack]]


class CodeGeneratorLoader:
  def __init__(self):
    self.domain2opset: Dict[str, OpSet] = {}
    self._register_builtin_code_generators()

  def register_code_generator(self, domain: Optional[str], op_name: str, pack: CodeGeneratorPack):
    if domain is None or domain == "":
      domain = "ai.onnx"
    if domain not in self.domain2opset:
      self.domain2opset[domain] = {}
    if op_name not in self.domain2opset[domain]:
      self.domain2opset[domain][op_name] = []
    self.domain2opset[domain][op_name].append(pack)
    self.domain2opset[domain][op_name].sort(key=lambda x: x.version, reverse=True)

  def register_from_py_file(self, filepath: Union[str, bytes, os.PathLike]) -> int:
    """
    Register code generator from a python file.

    Each python file could contain multiple code generator classes.
    Each class should have the following class attributes:
      * The field `__OP_NAME__` is the name of the operator. If not specified, the name of the class will be used.
      * The field `__OP_DOMAIN__` is the name of the operator. If not specified, default to `custom`.
      * The field `__OP_VERSION__` is the version of this operator. If not specified, the version will be set to 1.

    Parameters
    ----------
    domain : str
      The domain of the operator.
    filepath : str or bytes or os.PathLike
      The path to the python file.

    Returns
    -------
    int : The number of code generators registered from the file.
    """
    filepath = str(filepath)
    assert os.path.exists(filepath), f"{filepath} does not exist"
    assert os.path.isfile(filepath), f"{filepath} is not a file"
    assert filepath.endswith(".py"), f"{filepath} is not a .py file"
    # load module
    module_name = os.path.basename(filepath)[:-3]
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # register
    custom_loaded_count = 0
    for clazz_name, clazz in inspect.getmembers(module, inspect.isclass):
      if clazz_name == "CustomOpCodeGenerator" or not issubclass(clazz, CustomOpCodeGenerator):
        continue
      domain = getattr(clazz, "__OP_DOMAIN__", "custom")
      self.register_code_generator(
        domain,
        getattr(clazz, "__OP_NAME__", clazz_name),
        CodeGeneratorPack(
          version=getattr(clazz, "__OP_VERSION__", 1),
          module=module,
          class_name=clazz_name
        )
      )
      custom_loaded_count += 1
      logging.debug(f"Register code generator {clazz_name} for domain {domain} from {filepath}")
    return custom_loaded_count

  def get(self, domain: Optional[str], op_name: str, version: Optional[int] = None) -> Optional[OpCodeGenerator]:
    if domain is None or domain == "":
      domain = "ai.onnx"
    if domain in self.domain2opset:
      if op_name in self.domain2opset[domain]:
        if version is None:
          return self.domain2opset[domain][op_name][0].new()
        else:
          for pack in self.domain2opset[domain][op_name]:
            if version >= pack.version:
              return pack.new()
        return None
    return None

  def drop(self, domain: str, op_name: Optional[str] = None):
    if domain in self.domain2opset:
      if op_name is None:
        del self.domain2opset[domain]
      elif op_name in self.domain2opset[domain]:
        del self.domain2opset[domain][op_name]
      if len(self.domain2opset[domain]) == 0:
        del self.domain2opset[domain]

  def reset(self):
    self.domain2opset = {}

  def __len__(self):
    """
    Length of all registered code generators type.

    Returns
    -------
    int
    """
    return sum([len(ops) for ops in self.domain2opset.values()])

  def _register_builtin_code_generators(self):
    modules = inspect.getmembers(onnx_pytorch.op_code_generators, inspect.ismodule)
    builtin_loaded_count = 0
    for module_name, module in modules:
      module = getattr(onnx_pytorch.op_code_generators, module_name)
      clazz_name = f"{module_name}OpCodeGenerator"
      if hasattr(module, clazz_name):
        clazz = getattr(module, clazz_name)
        if inspect.isclass(clazz) and issubclass(clazz, OpCodeGenerator):
          # TODO: support multiple versions
          self.register_code_generator(None, module_name, CodeGeneratorPack(5, module, clazz_name))
          builtin_loaded_count += 1
          logging.debug(f"Register builtin code generator class {clazz_name} from module {module_name}")
    logging.info(f"Loaded {builtin_loaded_count} builtin code generators.")


# global instance
code_generator_loader = CodeGeneratorLoader()
