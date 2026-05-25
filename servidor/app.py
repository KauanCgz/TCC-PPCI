"""
Servidor PPCI — API REST
Hospeda os PDFs das INs do CBMSC e expõe um endpoint /consultar.
Os clientes enviam apenas a pergunta + própria API Key.
Os PDFs nunca saem do servidor.
"""

import os
import threading
import hashlib
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from anthropic import Anthropic

try:
    import pdfplumber
    _PDF_LIB = "pdfplumber"
except ImportError:
    _PDF_LIB = None

# ── Configuração ──────────────────────────────────────────────────────────
PDF_DIR   = Path(__file__).parent / "pdfs"
CACHE_DIR = Path(__file__).parent / "cache_texto"

# Token de acesso ao servidor (defina como variável de ambiente no Render)
# Se não definido, o servidor fica aberto (OK para protótipo de TCC)
SERVER_TOKEN = os.environ.get("SERVER_TOKEN", "")

app = FastAPI(
    title="API PPCI — CBMSC",
    description="Sistema multi-agente para consulta de normas de prevenção a incêndio.",
    version="2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base de conhecimento carregada na inicialização
pdf_textos: dict[str, str] = {}


# ── Extração de PDF ───────────────────────────────────────────────────────
def _hash_arquivo(caminho: str) -> str:
    return hashlib.md5(Path(caminho).read_bytes()).hexdigest()[:10]


def extrair_texto_pdf(caminho: str, nome: str) -> str:
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"{_hash_arquivo(caminho)}.txt"

    if cache.exists():
        return cache.read_text(encoding="utf-8")

    print(f"  Extraindo: {nome}...")
    paginas = []

    if _PDF_LIB == "pdfplumber":
        import pdfplumber
        with pdfplumber.open(caminho) as pdf:
            for i, pg in enumerate(pdf.pages):
                t = pg.extract_text()
                if t and t.strip():
                    paginas.append(f"[Pág. {i+1}]\n{t.strip()}")

    texto = "\n\n".join(paginas)
    cache.write_text(texto, encoding="utf-8")
    return texto


# ── Startup: carrega todos os PDFs ────────────────────────────────────────
@app.on_event("startup")
def carregar_pdfs():
    PDF_DIR.mkdir(exist_ok=True)
    pdfs = sorted(PDF_DIR.glob("*.pdf"))

    if not pdfs:
        print(f"⚠️  Nenhum PDF em {PDF_DIR}. Adicione as INs do CBMSC.")
        return

    print(f"\n📂 Carregando {len(pdfs)} PDF(s)...")
    for pdf in pdfs:
        nome = pdf.stem.replace("_", " ").replace("-", " ").strip()
        texto = extrair_texto_pdf(str(pdf), nome)
        if texto.strip():
            pdf_textos[nome] = texto
            print(f"  ✅ {nome} — {len(texto):,} chars")
        else:
            print(f"  ⚠️  {nome} — sem texto (PDF escaneado?)")

    print(f"\n🔥 {len(pdf_textos)} norma(s) disponíveis.\n")


# ── Autenticação opcional ─────────────────────────────────────────────────
token_header = APIKeyHeader(name="X-Server-Token", auto_error=False)

def verificar_token(token: str = Security(token_header)):
    if SERVER_TOKEN and token != SERVER_TOKEN:
        raise HTTPException(status_code=403, detail="Token de servidor inválido.")
    return token


# ── Schemas ───────────────────────────────────────────────────────────────
class ConsultaRequest(BaseModel):
    pergunta:    str
    api_key:     str   # chave Anthropic do próprio usuário

class ConsultaResponse(BaseModel):
    resposta:           str
    normas_consultadas: list[str]
    normas_com_info:    list[str]


# ── Endpoints ─────────────────────────────────────────────────────────────
@app.get("/")
def raiz():
    return {
        "servico": "API PPCI — CBMSC",
        "normas":  len(pdf_textos),
        "status":  "online",
    }


@app.get("/normas")
def listar_normas(_: str = Depends(verificar_token)):
    return {"normas": list(pdf_textos.keys()), "total": len(pdf_textos)}


@app.post("/consultar", response_model=ConsultaResponse)
def consultar(req: ConsultaRequest, _: str = Depends(verificar_token)):
    if not req.api_key.strip():
        raise HTTPException(400, "api_key não pode ser vazio.")
    if not req.pergunta.strip():
        raise HTTPException(400, "pergunta não pode ser vazia.")
    if not pdf_textos:
        raise HTTPException(503, "Nenhuma norma carregada no servidor.")

    try:
        client = Anthropic(api_key=req.api_key)
    except Exception as e:
        raise HTTPException(400, f"API Key inválida: {e}")

    # ── Consultar agentes funcionários em paralelo ────────────────────────
    respostas: list[dict] = []
    lock = threading.Lock()

    def consultar_agente(nome: str, texto: str):
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=600,
                system=(
                    f"Você é especialista EXCLUSIVAMENTE na norma: {nome}.\n"
                    "Responda SOMENTE com base no texto fornecido.\n"
                    "Se não houver informação relevante, responda: SEM_INFORMAÇÃO\n"
                    "Cite artigo, tabela ou seção quando possível."
                ),
                messages=[{
                    "role": "user",
                    "content": (
                        f"TEXTO DA NORMA — {nome}:\n{texto}\n\n"
                        f"{'─'*60}\nPERGUNTA: {req.pergunta}"
                    )
                }]
            )
            r = resp.content[0].text.strip()
            if "SEM_INFORMAÇÃO" not in r:
                with lock:
                    respostas.append({"agente": nome, "resposta": r})
        except Exception as e:
            print(f"Erro agente {nome}: {e}")

    threads = [
        threading.Thread(target=consultar_agente, args=(n, t), daemon=True)
        for n, t in pdf_textos.items()
    ]
    for t in threads: t.start()
    for t in threads: t.join()

    # ── Sem resultados ────────────────────────────────────────────────────
    if not respostas:
        return ConsultaResponse(
            resposta=(
                "⚠️  Nenhuma das normas carregadas contém informação sobre essa consulta.\n"
                f"Normas verificadas: {', '.join(pdf_textos.keys())}"
            ),
            normas_consultadas=list(pdf_textos.keys()),
            normas_com_info=[],
        )

    # ── Compilar resposta final ───────────────────────────────────────────
    contexto = "\n\n".join(
        f"╔══ {r['agente']} ══╗\n{r['resposta']}" for r in respostas
    )

    resp_final = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        system=(
            "Você é assistente técnico de projetista de PPCI (CBMSC).\n"
            "Use SOMENTE as informações abaixo. Cite a norma de origem. "
            "Se houver conflito entre normas, aponte."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Respostas das normas:\n\n{contexto}\n\n"
                f"{'═'*60}\nPergunta: {req.pergunta}\n\n"
                "Compile uma resposta clara para o projetista, citando as normas."
            )
        }]
    )

    return ConsultaResponse(
        resposta=resp_final.content[0].text.strip(),
        normas_consultadas=list(pdf_textos.keys()),
        normas_com_info=[r["agente"] for r in respostas],
    )
