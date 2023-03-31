import tml.reader.utils as reader_utils


def test_rr():
  options = ["a", "b", "c"]
  rr = reader_utils.roundrobin(options)
  for i, v in enumerate(rr):
    assert v == options[i % 3]
    if i > 4:
      break
