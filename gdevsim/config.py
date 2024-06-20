"""Store configuration."""

__all__ = ["PATH"]

import pathlib

home = pathlib.Path.home()
cwd = pathlib.Path.cwd()
cwd_config = cwd / "config.yml"

home_config = home / ".config" / "gdevsim.yml"
config_dir = home / ".config"
config_dir.mkdir(exist_ok=True)
module_path = pathlib.Path(__file__).parent.absolute()
repo_path = module_path.parent


class Path:
    module = module_path
    repo = repo_path
    temp = module_path / "temp"
    materials = module_path / "materials"
    simulation = repo_path / "simulation"
    ref_data = module_path / "ref_data"
    docs_examples = repo_path / "docs" / "examples"


PATH = Path()

if __name__ == "__main__":
    print(PATH)
