import inspect, os, shutil
import transformers

modeling_file = inspect.getfile(transformers.models.gemma4.modeling_gemma4)

src_path = r"./models/modeling_gemma4_standardAttention.py"

backup_folder = "./models_backup"
os.makedirs(backup_folder, exist_ok=True)

filename = os.path.basename(modeling_file)              # "modeling_gemma4.py"
backup_path = os.path.join(backup_folder, filename)

print(f"[1] located installed file : {modeling_file}")
if not os.path.exists(backup_path):
    print(f"[2] backing up             : {modeling_file} -> {backup_path}")
    shutil.copy2(modeling_file, backup_path)
else:
    print(f"[2] backup already exists  : {backup_path} (skipping)")
print(f"[3] overwriting installed  : {src_path} -> {modeling_file}")
shutil.copy2(src_path, modeling_file)
print("[done] installed modeling file replaced; restart the interpreter for it to take effect")
