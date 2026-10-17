"""Baixa o corpus público: OWASP Cheat Sheets, fixados em um commit específico.

Uso:  uv run python scripts/fetch_corpus.py          # baixa
      uv run python scripts/fetch_corpus.py --check  # confere os arquivos contra o manifesto

Por que um commit fixo? As métricas de avaliação só são comparáveis se o corpus for
sempre o mesmo. Atualizar o corpus é uma decisão explícita: troque OWASP_COMMIT,
rode o script e faça commit do MANIFEST.json novo.

Os .md baixados ficam fora do Git (.gitignore); o MANIFEST.json (commit + hash de cada
arquivo) é versionado e prova qual corpus gerou cada resultado.

Licença do corpus: CC BY-SA 4.0, © OWASP Foundation.
"""

import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

OWASP_REPO = "OWASP/CheatSheetSeries"
OWASP_COMMIT = "1a78ae3e443cef156e947f87e892efbcc62876ee"  # master em 2026-10-16

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "public" / "owasp"


def download_tarball(repo: str, commit: str) -> bytes:
    url = f"https://codeload.github.com/{repo}/tar.gz/{commit}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def extract_cheatsheets(tarball: bytes, dest: Path) -> dict[str, str]:
    """Extrai só `cheatsheets/*.md` e devolve {nome_do_arquivo: sha256}."""
    dest.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}

    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as tar:
        for member in tar.getmembers():
            # Caminho no tar: "<repo>-<commit>/cheatsheets/Nome.md"
            parts = Path(member.name).parts
            if not (member.isfile() and len(parts) == 3 and parts[1] == "cheatsheets"):
                continue
            if not parts[2].endswith(".md"):
                continue

            # Lemos o conteúdo e escrevemos nós mesmos, usando só o nome do arquivo.
            # Assim um caminho malicioso no tar (ex.: "../../x") nunca sai de `dest`.
            content = tar.extractfile(member).read()
            name = Path(parts[2]).name
            (dest / name).write_bytes(content)
            hashes[name] = hashlib.sha256(content).hexdigest()

    return dict(sorted(hashes.items()))


def check(dest: Path) -> int:
    """Confere se os arquivos locais são exatamente os do manifesto (nada faltando ou alterado)."""
    manifest = json.loads((dest / "MANIFEST.json").read_text(encoding="utf-8"))
    problems = []
    for name, expected in manifest["files"].items():
        path = dest / name
        if not path.exists():
            problems.append(f"faltando: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            problems.append(f"alterado: {name}")
    for problem in problems:
        print(problem)
    print("Corpus íntegro." if not problems else f"{len(problems)} problema(s).")
    return 1 if problems else 0


def main() -> None:
    if "--check" in sys.argv:
        sys.exit(check(DEST))

    print(f"Baixando {OWASP_REPO}@{OWASP_COMMIT[:7]}...")
    hashes = extract_cheatsheets(download_tarball(OWASP_REPO, OWASP_COMMIT), DEST)

    manifest = {
        "source": f"https://github.com/{OWASP_REPO}",
        "commit": OWASP_COMMIT,
        "license": "CC-BY-SA-4.0",
        "files": hashes,
    }
    (DEST / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{len(hashes)} cheat sheets em {DEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
