import inspect
import logging
from types import ModuleType
from typing import List, Tuple, Dict, Any, Optional, NamedTuple

import onnx_pytorch.op_code_generators
from onnx_pytorch.op_code_generators import OpCodeGenerator


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

  def drop(self, domain: str, op_name: str):
    if domain in self.domain2opset:
      if op_name in self.domain2opset[domain]:
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
      clazz_name = f"{module_name}OpCodeGenerator"
      if hasattr(module, clazz_name):
        clazz = getattr(module, clazz_name)
        if inspect.isclass(clazz) and issubclass(clazz, OpCodeGenerator):
          # TODO: support multiple versions
          self.register_code_generator(None, module_name, CodeGeneratorPack(5, module, clazz_name))
          builtin_loaded_count += 1
          logging.debug(f"Register code generator class {clazz_name} from module {module_name}")
    logging.info(f"Loaded {builtin_loaded_count} builtin code generators.")


# global instance
code_generator_loader = CodeGeneratorLoader()
