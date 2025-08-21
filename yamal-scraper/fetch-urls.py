import os
import yaml

OPENAPI_DIR = "openapi-yaml"
base_urls = {}

for file_name in os.listdir(OPENAPI_DIR):
    if not file_name.endswith(".yaml"):
        continue

    file_path = os.path.join(OPENAPI_DIR, file_name)
    try:
        with open(file_path, "r") as f:
            spec = yaml.safe_load(f)

        if not spec or "servers" not in spec:
            print(f"⚠️ No servers key in {file_name}")
            continue

        # Use first server as default
        server_url = spec["servers"][0]["url"]
        base_urls[file_name] = server_url
        print(f"✅ {file_name}: {server_url}")
    except Exception as e:
        print(f"❌ Error reading {file_name}: {e}")

print("\nBase URLs loaded:", base_urls)