"""Code location and explicit read-only legacy resource locations are separate."""
import json
import os
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[2]

def resources():
    path=Path(os.environ.get('HUMAN_DATA_RESOURCES',PROJECT_ROOT/'config/resources.json')).resolve()
    data=json.loads(path.read_text())
    if data.get('schema')!='human_data_resources_v2':raise ValueError('invalid_resources_schema')
    data['config_path']=str(path)
    return data

def resource_path(key):
    return Path(resources()[key]).resolve()
