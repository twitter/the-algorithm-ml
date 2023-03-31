"""
Simple str.split() parsing of input string

usage example:
  python list_ops.py --input_list=$INPUT [--sep=","] [--op=<len|select>] [--elem=$INDEX]

Args:
  - input_list: input string
  - sep (default ","): separator string
  - elem (default 0): integer index
  - op (default "select"): either `len` or `select`
    - len: prints len(input_list.split(sep))
    - select: prints input_list.split(sep)[elem]

Typical usage would be in a bash script, e.g.:

  LIST_LEN=$(python list_ops.py --input_list=$INPUT --op=len)

"""
import tml.machines.environment as env

from absl import app, flags


FLAGS = flags.FLAGS
flags.DEFINE_string("input_list", None, "string to parse as list")
flags.DEFINE_integer("elem", 0, "which element to take")
flags.DEFINE_string("sep", ",", "separator")
flags.DEFINE_string("op", "select", "operation to do")


def main(argv):
  split_list = FLAGS.input_list.split(FLAGS.sep)
  if FLAGS.op == "select":
    print(split_list[FLAGS.elem], flush=True)
  elif FLAGS.op == "len":
    print(len(split_list), flush=True)
  else:
    raise ValueError(f"operation {FLAGS.op} not recognized.")


if __name__ == "__main__":
  app.run(main)
