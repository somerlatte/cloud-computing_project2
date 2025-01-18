from flask import Flask, request, jsonify
import pickle
import os
import logging
from datetime import datetime

# Configurando logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

app = Flask(__name__)

# Caminho do modelo definido como variável de ambiente
MODEL_PATH = os.getenv("MODEL_PATH", "/mnt/shared/recommendation_rules.pkl")
MODEL_METADATA = {"last_updated": None}

def carregar_modelo(caminho_modelo):
    """Carrega o modelo de recomendação a partir de um arquivo."""
    if not os.path.isfile(caminho_modelo):
        logging.error(f"Arquivo do modelo não encontrado: {caminho_modelo}")
        raise FileNotFoundError(f"Arquivo do modelo não encontrado: {caminho_modelo}")
    
    # Captura a última data de modificação do arquivo
    MODEL_METADATA["last_updated"] = datetime.fromtimestamp(
        os.path.getmtime(caminho_modelo)
    ).strftime("%Y-%m-%d %H:%M:%S")

    with open(caminho_modelo, "rb") as arquivo:
        modelo = pickle.load(arquivo)

    logging.info(f"Modelo carregado com sucesso. Última atualização: {MODEL_METADATA['last_updated']}")
    return modelo

def gerar_recomendacoes(musicas_usuario, regras, limite_top=5, max_regras=1000):
    """Gera recomendações de playlists com base nas regras e músicas fornecidas."""
    recomendacoes = []
    contador_regras = 0

    for regra in regras:
        if contador_regras >= max_regras:
            break

        antecedentes = set(regra.get("antecedent", []))
        consequentes = set(regra.get("consequent", []))
        confianca = regra.get("confidence", 0)

        if antecedentes.issubset(musicas_usuario):
            recomendacoes.append({"songs": list(consequentes), "confidence": confianca})
            contador_regras += 1

    # Ordena as recomendações pela confiança de forma decrescente
    return sorted(recomendacoes, key=lambda x: x["confidence"], reverse=True)[:limite_top]

@app.route("/api/recommend", methods=["POST"])
def endpoint_recomendacao():
    """Endpoint para recomendar músicas com base nas regras carregadas."""
    try:
        dados = request.get_json()

        if not dados or "songs" not in dados:
            return jsonify({"error": "Dados inválidos ou campo 'songs' ausente"}), 400

        musicas_usuario = set(dados["songs"])
        if not musicas_usuario:
            return jsonify({"error": "A lista de músicas não pode estar vazia"}), 400

        recomendacoes = gerar_recomendacoes(musicas_usuario, app.regras, limite_top=5)

        resposta = {
            "recommendations": recomendacoes,
            "version": app.versao,
            "model_last_updated": MODEL_METADATA["last_updated"]
        }
        return jsonify(resposta), 200
    except Exception as e:
        logging.error(f"Erro ao processar a requisição: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    try:
        # Carrega o modelo e inicializa a aplicação
        app.regras = carregar_modelo(MODEL_PATH)
        app.versao = "2.0.0"
        app.run(host="0.0.0.0", port=52041)
    except Exception as erro:
        logging.critical(f"Falha ao iniciar a aplicação: {erro}")
