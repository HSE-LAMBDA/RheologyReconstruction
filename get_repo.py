import os
os.system("git clone https://github.com/HSE-LAMBDA/RheologyReconstruction.git")
os.system("git config --global user.email \"you@example.com\"")
os.system("git config --global user.name \"Your Name\"")
os.system("cd RheologyReconstruction; git pull origin dev; cd ../")
subfolders = ['dolfin_adjoint', 'neural_networks']
mod_names = ['dataset', 'logger', 'metrics', 'trainer', 'transforms', 'utils', 'trainer']
filenames = [f"RheologyReconstruction/pipeline/{mn}.py" for mn in mod_names] 

for sf in subfolders:
  for mn in mod_names:
    p1 = f"RheologyReconstruction/pipeline/{mn}.py"
    p2 = f"RheologyReconstruction/pipeline/{sf}/{mn}.py"
    print(os.system(f"cp {p1} {p2}"))


for sf in subfolders:
  new_path = os.path.join('RheologyReconstruction/pipeline', sf)
  for fn in os.listdir(new_path):
    if fn[-3:] == '.py':
      filenames.append(os.path.join(new_path, fn))
print(filenames)

for filename in filenames:
  with open(filename, "r") as f:
    lines = f.readlines()

  for i in range(len(lines)):
    line = lines[i]
    if line[:4] == 'from':
      l = line.split()
      mod_names = ['dataset', 'logger', 'metrics', 'trainer', 'transforms', 'utils', 'trainer']
      if l[1] in mod_names or l[1].split('.')[0] in subfolders:
        l[1] = '.' + l[1]
      line = ' '.join(l)
      #lines[10] = 'from .dataset import SeismogramBatch\n'
      lines[i] = line + '\n'

  with open(filename, "w") as f:
    f.writelines(lines)
    
