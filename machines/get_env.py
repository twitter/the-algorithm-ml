import tml.machines.environment as env

from absl import app, flags


FLAGS = flags.FLAGS
flags.DEFINE_string("property", None, "Which property of the current environment to fetch.")


def main(argv):
  if FLAGS.property == "using_dds":
    print(f"{env.has_readers()}", flush=True)
  if FLAGS.property == "has_readers":
    print(f"{env.has_readers()}", flush=True)
  elif FLAGS.property == "get_task_type":
    print(f"{env.get_task_type()}", flush=True)
  elif FLAGS.property == "is_datasetworker":
    print(f"{env.is_reader()}", flush=True)
  elif FLAGS.property == "is_dds_dispatcher":
    print(f"{env.is_dispatcher()}", flush=True)
  elif FLAGS.property == "get_task_index":
    print(f"{env.get_task_index()}", flush=True)
  elif FLAGS.property == "get_dataset_service":
    print(f"{env.get_dds()}", flush=True)
  elif FLAGS.property == "get_dds_dispatcher_address":
    print(f"{env.get_dds_dispatcher_address()}", flush=True)
  elif FLAGS.property == "get_dds_worker_address":
    print(f"{env.get_dds_worker_address()}", flush=True)
  elif FLAGS.property == "get_dds_port":
    print(f"{env.get_reader_port()}", flush=True)
  elif FLAGS.property == "get_dds_journaling_dir":
    print(f"{env.get_dds_journaling_dir()}", flush=True)
  elif FLAGS.property == "should_start_dds":
    print(env.is_reader() or env.is_dispatcher(), flush=True)


if __name__ == "__main__":
  app.run(main)
