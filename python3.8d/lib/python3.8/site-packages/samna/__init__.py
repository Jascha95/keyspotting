import subprocess
import sys
from importlib.metadata import version

PYPI_SAMNA = True  # a mark to indicate it's pypi samna

__gitlab_packages_index_url__ = (
    "https://gitlab.com/api/v4/projects/27423070/packages/pypi/simple"
)
__version__ = version("samna")


def __install_real_samna__(index_url, version):
    # Currently pypi samna is already installed, so old samna is already uninstalled
    # We just need to force reinstall samna to use real samna instead of pypi samna.
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            f"samna=={version}",
            f"--index-url={index_url}",
        ]
    )


def install_real_samna():
    __install_real_samna__(__gitlab_packages_index_url__, __version__)


# install real samna of the same version of current pypi samna.
install_real_samna()

# import real samna to this samna namespace
import samna  # this samna is current cached pypi samna
import importlib

# force to reload the real samna, samna imported after this will use this cache.
importlib.reload(samna)
