import inspect, os, shutil
import transformers

modeling_file = inspect.getfile(transformers.models.gemma4.modeling_gemma4)
backup_folder = "./models_backup"

filename = os.path.basename(modeling_file)          # "modeling_gemma4.py"
backup_path = os.path.join(backup_folder, filename)

if not os.path.exists(backup_path):
    raise FileNotFoundError(f"No backup found at {backup_path} — nothing to restore from. Please reinstall package.\npip install transformers==5.5.4")

print(f"[1] backup found           : {backup_path}")
print(f"[2] restoring              : {backup_path} -> {modeling_file}")
shutil.copy2(backup_path, modeling_file)
print("[done] original modeling file restored; restart the interpreter for it to take effect")