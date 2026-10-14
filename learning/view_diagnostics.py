"""Open a diagnostic bundle without copying over rscope's active-run directory.

On macOS: mjpython -m learning.view_diagnostics /path/to/diagnostics/final
"""
import argparse
from pathlib import Path


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('bundle', type=Path)
  args = parser.parse_args()
  import rscope.config as config
  from rscope.main import main as view
  directory = (args.bundle / 'rscope').resolve()
  if not (directory / 'rscope_meta.pkl').is_file():
    parser.error(f'No rscope metadata in {directory}')
  config.BASE_PATH = directory
  config.META_PATH = directory / 'rscope_meta.pkl'
  view()


if __name__ == '__main__':
  main()
