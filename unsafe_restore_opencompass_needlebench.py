import inspect, os, shutil
import opencompass

root_folder = os.path.dirname(inspect.getfile(opencompass))
dst_folder = os.path.join(root_folder, r"configs/datasets/needlebench_v2/needlebench_v2_4k")
print(f"[1] located needlebench_v2_4k folder : {dst_folder}")

backup_folder = r"./models_backup"

for i, filename in enumerate([r"needlebench_v2_4k.py", r"needlebench_v2_single_4k.py"]):
    dst_path = os.path.join(dst_folder, filename)
    backup_path = os.path.join(backup_folder, filename)

    if os.path.exists(backup_path):
        print(f"[{i + 2}.2] copying  : {backup_path} -> {dst_path}")
        shutil.copy2(backup_path, dst_path)
    else:
        print(f"[{i+2}.1] backup doesn't exists  : {backup_path} (skipping)")

print("[done] config restored")
