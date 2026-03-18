import re, pathlib, json

def load_yaml(path):
    txt = pathlib.Path(path).read_text()
    txt = re.sub(r"#.*", "", txt)                       # remove comments
    data = {}
    for line in txt.splitlines():
        line = line.strip()
        if not line or ':' not in line:
            continue
        key, val = line.split(':', 1)
        key, val = key.strip(), val.strip()
        # ------  strip optional quotes  ------
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        # ------  lists  ------
        if val.startswith('[') and val.endswith(']'):
            val = json.loads(val)
        # ------  numbers  ------
        elif val.replace('.', '', 1).isdigit():
            val = float(val) if '.' in val else int(val)
        # ------  booleans  ------
        elif val.lower() in ('true', 'false'):
            val = val.lower() == 'true'
        data[key] = val
    return data
