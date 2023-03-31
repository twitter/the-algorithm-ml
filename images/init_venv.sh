#! /bin/sh

if [[ "$(uname)" == "Darwin" ]]; then
  echo "Only supported on Linux."
  exit 1
fi

# You may need to point this to a version of python 3.10
PYTHONBIN="/opt/ee/python/3.10/bin/python3.10"
echo Using "PYTHONBIN=$PYTHONBIN"

# Put venv in tmp, these things are not made to last, just rebuild.
VENV_PATH="$HOME/tml_venv"
rm -rf "$VENV_PATH"
"$PYTHONBIN" -m venv "$VENV_PATH"

# shellcheck source=/dev/null
. "$VENV_PATH/bin/activate"

pip --require-virtual install -U pip
pip --require-virtualenv install --no-deps -r images/requirements.txt

ln -s "$(pwd)" "$VENV_PATH/lib/python3.10/site-packages/tml"

echo "Now run source ${VENV_PATH}/bin/activate" to get going.
