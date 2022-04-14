"""
Helper functions for ONNX attributes handling in code generation.
"""
from typing import List, Optional, Tuple, Union, Dict, Any


def pluralize_single_item(item: Union[Any, List[Any], Tuple[Any]], target_size: int = 2) -> List[Any]:
  """
  Converts a single item to a list of size target_size.

  Parameters
  ----------
  item
  target_size : int
    The target size of the returned list.

  Returns
  -------
  list
  """
  if isinstance(item, (list, tuple)):
    assert len(item) == 1
    element = item[0]
    return [element] * target_size
  else:
    return [item] * target_size
