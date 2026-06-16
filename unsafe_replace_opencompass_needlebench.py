import inspect, os, shutil
import opencompass

root_folder = os.path.dirname(inspect.getfile(opencompass))
dst_folder = os.path.join(root_folder, r"configs/datasets/needlebench_v2")
print(f"[1] located needlebench_v2 folder : {dst_folder}")

src_folder = r"./models"
backup_folder = r"./models_backup"
os.makedirs(backup_folder, exist_ok=True)

for i, rel_filename in enumerate([
    r"needlebench_v2_4k/needlebench_v2_4k.py",
    r"needlebench_v2_4k/needlebench_v2_single_4k.py",
    r"needlebench_v2_32k/needlebench_v2_32k.py",
    r"needlebench_v2_32k/needlebench_v2_single_32k.py",
    r"needlebench_v2_128k/needlebench_v2_128k.py",
    r"needlebench_v2_128k/needlebench_v2_single_128k.py",
]):
    filename = os.path.basename(rel_filename)
    src_path = os.path.join(src_folder, filename)
    dst_path = os.path.join(dst_folder, rel_filename)
    backup_path = os.path.join(backup_folder, filename)

    if not os.path.exists(backup_path):
        print(f"[{i+2}.1] backing up             : {dst_path} -> {backup_path}")
        shutil.copy2(dst_path, backup_path)
    else:
        print(f"[{i+2}.1] backup already exists  : {backup_path} (skipping)")

    print(f"[{i+2}.2] copying  : {src_path} -> {dst_path}")
    shutil.copy2(src_path, dst_path)

print("[done] config installed")
