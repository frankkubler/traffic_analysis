## Accélération GPU Intel (OpenVINO)

L'inférence peut être déportée sur un GPU intégré Intel via le backend OpenVINO
d'Ultralytics. Sur les iGPU Gen9 et supérieures (HD Graphics 6xx, UHD, Iris Xe),
le gain vient surtout du FP16, natif sur ce matériel.

Le paquet Python `openvino` embarque le plugin GPU, **mais pas le pilote**.
Sans le runtime OpenCL installé au niveau système, OpenVINO ne verra que le CPU
et l'inférence retombera silencieusement dessus.

### Linux (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install -y intel-opencl-icd clinfo
```

Vérification :

```bash
clinfo -l
# Platform #0: Intel(R) OpenCL Graphics
#  `-- Device #0: Intel(R) HD Graphics 620

python -c "import openvino as ov; print(ov.Core().available_devices)"
# ['CPU', 'GPU']
```

Si `available_devices` ne renvoie que `['CPU']` alors que `clinfo -l` liste bien
le GPU, vérifier l'accès au device node :

```bash
ls -l /dev/dri/renderD128     # le '+' indique une ACL logind active
id | grep -o render           # sinon : sudo usermod -aG render $USER puis reconnexion
```

Les dépôts Ubuntu embarquent une version de NEO (le runtime compute Intel)
qui a souvent un ou deux ans de retard. Pour un pilote plus récent, utiliser
le dépôt Intel :

```bash
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key \
  | sudo gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
  https://repositories.intel.com/gpu/ubuntu $(lsb_release -cs) client" \
  | sudo tee /etc/apt/sources.list.d/intel-gpu.list
sudo apt update && sudo apt install -y libze-intel-gpu1 libze1 intel-opencl-icd
```

### Windows (non testé)

Le runtime OpenCL est fourni directement par le pilote graphique Intel — il n'y a
rien à installer séparément.

1. Installer le pilote **Intel Arc & Iris Xe Graphics** (ou **Intel Graphics –
   Windows DCH**) depuis le centre de téléchargement Intel, ou via l'Intel Driver
   & Support Assistant. Les pilotes fournis par Windows Update ou par le
   constructeur du PC sont souvent trop anciens pour OpenVINO.
2. Redémarrer.
3. Vérifier depuis l'environnement Python du projet :

```powershell
python -c "import openvino as ov; print(ov.Core().available_devices)"
```

Si le GPU n'apparaît pas, contrôler la version du pilote dans le Gestionnaire de
périphériques : OpenVINO demande une build relativement récente (30.0.101.x ou
supérieure sur les générations récentes).

### Utilisation

Le modèle doit être exporté au format OpenVINO **avant** de pouvoir cibler le GPU.
Sur un fichier `.pt`, Ultralytics passe par PyTorch et ignore la cible Intel :

```python
from ultralytics import YOLO

# Une seule fois — produit le dossier yolov8n_openvino_model/
YOLO("yolov8n.pt").export(format="openvino", half=True)

model = YOLO("yolov8n_openvino_model/", task="detect")
results = model(frame, device="intel:gpu")
```

`half=True` produit un IR en FP16 : c'est le principal levier de performance sur
iGPU. L'INT8 n'apporte pas grand-chose avant Gen12 (pas de DP4a sur Gen9).

Le reste du pipeline supervision est inchangé :

```python
detections = sv.Detections.from_ultralytics(results[0])
```

### Vérifier que le GPU travaille vraiment

```bash
sudo apt install -y intel-gpu-tools
sudo intel_gpu_top
```

La ligne *Render/3D* doit monter pendant le traitement. Si elle reste à zéro,
l'inférence tourne sur le CPU malgré `device="intel:gpu"` — dans la quasi-totalité
des cas, c'est que le modèle chargé est un `.pt` et non le dossier
`*_openvino_model/`.

### Dépannage

| Symptôme | Cause probable |
|---|---|
| `clinfo -l` ne renvoie rien | `intel-opencl-icd` absent |
| `available_devices` → `['CPU']` seul | pilote absent, ou utilisateur hors du groupe `render` |
| Premier appel très lent (20–40 s) | compilation des kernels ; le cache OpenVINO la supprime aux runs suivants |
| Aucun gain face au CPU | modèle exporté en FP32 — réexporter avec `half=True` |