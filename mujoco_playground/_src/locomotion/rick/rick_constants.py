import subprocess
from etils import epath

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "rick"

def ensure_rick_assets():
    xml_dir = ROOT_PATH / "xmls"
    rick_v2_dir = xml_dir / "rick_v2"
    if not rick_v2_dir.exists():
        xml_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cloning rick_v2 repository into {rick_v2_dir}...")
        subprocess.run(
            ["git", "clone", "https://github.com/badfortrains/rick_v2.git", str(rick_v2_dir)],
            check=True,
        )

def task_to_xml(task_name: str) -> epath.Path:
    ensure_rick_assets()
    xml_path = ROOT_PATH / "xmls" / "rick_v2" / "v3Robot_v18.xml"
    return xml_path
