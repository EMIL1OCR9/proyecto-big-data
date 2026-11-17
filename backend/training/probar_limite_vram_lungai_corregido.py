"""LungAI: prueba aislada de VRAM para ResNet-10 3D, sin CT ni escritura de modelos.

Ejecutar desde ~/lungai-venv:
    python probar_limite_vram_lungai.py

Cada tamaño se prueba en un proceso independiente. No determina por sí solo
la resolución clínicamente útil ni garantiza el éxito de un entrenamiento completo.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

TAMANOS = [
    (128, 128, 112),
    (144, 144, 128),
    (160, 160, 128),
    (160, 160, 144),
    (176, 176, 144),
    (176, 176, 160),
    (192, 192, 160),
    (192, 192, 176),
    (208, 208, 176),
    (224, 224, 192),
]


def probar_hijo(xyz: tuple[int, int, int]) -> None:
    import torch
    from monai.networks.nets import resnet10

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: no hay GPU CUDA disponible en este entorno.")
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.get_device_name(0)
    torch.cuda.reset_peak_memory_stats()
    total = torch.cuda.get_device_properties(0).total_memory
    libre_inicio, _ = torch.cuda.mem_get_info()
    try:
        modelo = resnet10(spatial_dims=3, n_input_channels=1, num_classes=2).to(device)
        optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-4)
        entrada = torch.randn(1, 1, xyz[2], xyz[1], xyz[0], device=device)
        etiqueta = torch.tensor([1], dtype=torch.long, device=device)
        modelo.train()
        for _ in range(2):
            optimizador.zero_grad(set_to_none=True)
            logits = modelo(entrada)
            perdida = torch.nn.functional.cross_entropy(logits, etiqueta)
            perdida.backward()
            optimizador.step()
        modelo.eval()
        with torch.inference_mode():
            _ = modelo(entrada)
        torch.cuda.synchronize()
        pico = torch.cuda.max_memory_reserved()
        libre_final, _ = torch.cuda.mem_get_info()
        print("MEMORY_RESULT " + json.dumps({
            "size": list(xyz),
            "peak_gb": round(pico / 2**30, 3),
            "total_gb": round(total / 2**30, 3),
            "free_at_start_gb": round(libre_inicio / 2**30, 3),
            "free_after_gb": round(libre_final / 2**30, 3),
        }), flush=True)
    except torch.cuda.OutOfMemoryError:
        print("MEMORY_OOM", flush=True)
        raise SystemExit(12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=int, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--max-usage", type=float, default=0.78,
                        help="Fraccion maxima conservadora de VRAM total (default 0.78)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Segundos maximos por tamaño (default 120)")
    args = parser.parse_args()
    if args.child:
        probar_hijo(tuple(args.child))
        return
    if not 0.5 <= args.max_usage <= 0.90:
        raise SystemExit("Usa --max-usage entre 0.50 y 0.90")
    print("LUNGAI: BUSQUEDA PROGRESIVA DE VRAM (DATOS SINTETICOS)", flush=True)
    print("Red: ResNet-10 3D; batch 1; FP32; Adam; forward/backward x2.", flush=True)
    print("No se leen CT, no se entrena en pacientes, no se escriben archivos.", flush=True)
    print("El límite conservador es %.0f%% de VRAM.\n" % (args.max_usage * 100), flush=True)
    recomendado = None
    maximo_pasado = None
    for xyz in TAMANOS:
        tam = "x".join(map(str, xyz))
        print(f"Probando {tam}...", flush=True)
        try:
            p = subprocess.run(
                [sys.executable, __file__, "--child", *map(str, xyz)],
                capture_output=True, text=True, timeout=args.timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            print(f"  Tiempo límite superado ({args.timeout} s). Se detiene la búsqueda.", flush=True)
            break
        salida = next((s.split("MEMORY_RESULT ", 1)[1]
                       for s in p.stdout.splitlines() if s.startswith("MEMORY_RESULT ")), None)
        if salida is not None and p.returncode == 0:
            r = json.loads(salida)
            maximo_pasado = xyz
            pico = r["peak_gb"]
            ratio = pico / r["total_gb"]
            libre = r["free_after_gb"]
            print(f"  OK: {pico:.2f}/{r['total_gb']:.2f} GiB de VRAM reservada; "
                  f"libre al finalizar: {libre:.2f} GiB", flush=True)
            if ratio <= args.max_usage and libre >= 0.65:
                recomendado = xyz
            else:
                print("  Sin margen seguro para entrenamiento real; se detiene.", flush=True)
                break
        elif p.returncode == 12 or "out of memory" in (p.stdout+p.stderr).lower():
            print("  CUDA OUT OF MEMORY: se detiene. No repetir este tamaño.", flush=True)
            break
        else:
            print("  Fallo diferente de OOM; se detiene.", flush=True)
            for linea in (p.stdout + '\n' + p.stderr).splitlines()[-12:]:
                print("  " + linea, flush=True)
            break
    print("\nRESULTADO DE PRUEBA", flush=True)
    print("Mayor tamaño que pasó:", maximo_pasado or "ninguno", flush=True)
    print("Tamaño recomendado con margen:", recomendado or "ninguno", flush=True)
    print("ADVERTENCIA: una CT real, otros procesos y mayor duración pueden consumir más RAM/VRAM.", flush=True)
    print("No ejecutar entrenamiento en el tamaño máximo solo por haber pasado esta prueba.", flush=True)


if __name__ == "__main__":
    main()
