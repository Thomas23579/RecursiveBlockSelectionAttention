import inspect, os, shutil

from opencompass.configs.models.gemma import hf_gemma2_2b_it
dst_folder = os.path.dirname(inspect.getfile(hf_gemma2_2b_it))
print(f"[1] located models folder : {dst_folder}")

src_folder = r"./models"

for i, filename in enumerate([r"hf_gemma4_e2b_it.py", r"hf_gemma4_e4b_it.py", r"hf_gemma4_31B_it.py"]):
    src_path = os.path.join(src_folder, filename)
    dst_path = os.path.join(dst_folder, filename)

    print(f"[{i+2}] copying  : {src_path} -> {dst_path}")
    shutil.copy2(src_path, dst_path)

print("[done] models installed")