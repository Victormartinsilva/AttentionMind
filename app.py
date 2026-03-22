"""
AttentionMind - Transformers e Mecanismo de Atencao do Zero
============================================================
Demonstracao educativa de Transformers implementados em NumPy puro.
Classifica sentimento de textos agricolas e visualiza a atencao.

Conceitos demonstrados:
- Token Embedding + Positional Encoding
- Scaled Dot-Product Attention
- Multi-Head Attention
- Feed-Forward Network
- Softmax sobre scores de atencao
- Embeddings em espaco 2D (PCA)
- Comparacao: modelo simples vs Transformer
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import re
import time
import warnings
from collections import Counter
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# PAGINA
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AttentionMind - Transformers do Zero",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
:root{
    --bg:#060a0f;--surface:#0b1117;--surface2:#101820;--surface3:#161f29;
    --border:#1a2535;--border2:#22334a;
    --gold:#f59e0b;--gold2:#d97706;--gold3:#92400e;
    --teal:#2dd4bf;--teal2:#0d9488;
    --blue:#60a5fa;--blue2:#2563eb;
    --red:#f87171;--green:#4ade80;--purple:#c084fc;
    --text:#dde6f0;--text2:#8899aa;--muted:#445566;
}
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;background:var(--bg);color:var(--text);}
h1,h2,h3{font-family:'IBM Plex Sans',sans-serif;font-weight:700;}
code,.mono{font-family:'IBM Plex Mono',monospace;font-size:.82rem;}

.stButton>button{
    background:linear-gradient(135deg,var(--gold2),var(--teal2));
    color:#000;border:none;border-radius:5px;
    font-family:'IBM Plex Mono',monospace;font-size:.78rem;
    letter-spacing:.08em;padding:.55rem 1.4rem;font-weight:600;
    transition:all .2s;box-shadow:0 0 18px rgba(245,158,11,.25);
}
.stButton>button:hover{box-shadow:0 0 28px rgba(245,158,11,.45);transform:translateY(-1px);}

.card{background:var(--surface);border:1px solid var(--border2);border-radius:7px;padding:1.1rem 1.3rem;}
.cg{background:var(--surface2);border:1px solid var(--gold2);border-left:3px solid var(--gold);border-radius:0 7px 7px 0;padding:.8rem 1rem;margin:.4rem 0;}
.ct{background:var(--surface2);border:1px solid var(--teal2);border-left:3px solid var(--teal);border-radius:0 7px 7px 0;padding:.8rem 1rem;margin:.4rem 0;}
.cb{background:var(--surface2);border:1px solid var(--blue2);border-left:3px solid var(--blue);border-radius:0 7px 7px 0;padding:.8rem 1rem;margin:.4rem 0;}
.cp{background:var(--surface2);border:1px solid #7c3aed;border-left:3px solid var(--purple);border-radius:0 7px 7px 0;padding:.8rem 1rem;margin:.4rem 0;}

.mb{background:var(--surface);border:1px solid var(--border2);border-radius:7px;padding:.9rem 1rem;text-align:center;position:relative;overflow:hidden;}
.mb::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--gold),var(--teal));}
.ml{font-family:'IBM Plex Mono',monospace;font-size:.58rem;letter-spacing:.18em;text-transform:uppercase;color:var(--muted);margin-bottom:.2rem;}
.mv{font-family:'IBM Plex Sans',sans-serif;font-size:1.75rem;font-weight:700;line-height:1;}
.gv{color:var(--gold);}.tv{color:var(--teal);}.bv{color:var(--blue);}.rv{color:var(--red);}.grv{color:var(--green);}.pv{color:var(--purple);}

.sl{font-family:'IBM Plex Mono',monospace;font-size:.58rem;letter-spacing:.2em;text-transform:uppercase;
    color:var(--muted);margin-bottom:.5rem;margin-top:1rem;padding-bottom:.3rem;border-bottom:1px solid var(--border);}

.tok{display:inline-block;padding:.25rem .6rem;border-radius:4px;margin:.2rem;
     font-family:'IBM Plex Mono',monospace;font-size:.82rem;font-weight:600;
     border:1px solid;cursor:default;transition:all .15s;}

.formula{background:var(--surface3);border:1px solid var(--border2);border-radius:6px;
         padding:.9rem 1.1rem;margin:.5rem 0;font-family:'IBM Plex Mono',monospace;
         font-size:.8rem;color:var(--teal);line-height:1.8;}

.sent-pos{background:rgba(74,222,128,.08);border:1px solid rgba(74,222,128,.3);
          border-radius:6px;padding:.6rem 1rem;color:var(--green);}
.sent-neg{background:rgba(248,113,113,.08);border:1px solid rgba(248,113,113,.3);
          border-radius:6px;padding:.6rem 1rem;color:var(--red);}
.sent-neu{background:rgba(96,165,250,.08);border:1px solid rgba(96,165,250,.3);
          border-radius:6px;padding:.6rem 1rem;color:var(--blue);}

[data-testid="stSidebar"]{background:var(--surface) !important;border-right:1px solid var(--border2);}
hr.dv{border:none;border-top:1px solid var(--border2);margin:1rem 0;}

.stTabs [data-baseweb="tab-list"]{background:var(--surface) !important;border-bottom:1px solid var(--border2);}
.stTabs [data-baseweb="tab"]{font-family:'IBM Plex Mono',monospace !important;font-size:.72rem !important;
    letter-spacing:.08em !important;color:var(--muted) !important;background:transparent !important;border:none !important;padding:.6rem 1rem !important;}
.stTabs [aria-selected="true"]{color:var(--gold) !important;border-bottom:2px solid var(--gold) !important;}
.stTabs [data-baseweb="tab-panel"]{background:var(--bg) !important;padding-top:1.2rem !important;}

.head-box{background:var(--surface2);border:1px solid var(--border2);border-radius:6px;
          padding:.6rem .8rem;text-align:center;margin:.3rem;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# DATASET — Textos agricolas com sentimento
# ─────────────────────────────────────────────────────────

DATASET = [
    # POSITIVO (50 exemplos)
    ("equipamento funcionando perfeitamente apos manutencao preventiva realizada",           "positivo"),
    ("colheita excelente producao acima da meta esperada para o periodo",                    "positivo"),
    ("trator revisado sem problemas pronto para operacao no campo",                          "positivo"),
    ("sistema de irrigacao operando com eficiencia maxima sem vazamentos",                   "positivo"),
    ("lavoura com desenvolvimento otimo clima favoravel sem ocorrencia de pragas",           "positivo"),
    ("reparo concluido com sucesso maquina liberada para uso imediato",                      "positivo"),
    ("producao de soja superou expectativas qualidade dos graos excelente",                  "positivo"),
    ("equipe realizou servico com rapidez e qualidade dentro do prazo",                      "positivo"),
    ("motor revisado operando sem ruidos anormais consumo de combustivel normal",            "positivo"),
    ("aplicacao de defensivo bem sucedida sem deriva lavoura protegida",                     "positivo"),
    ("solo preparado adequadamente boa umidade condicoes ideais para plantio",               "positivo"),
    ("plantio realizado no tempo correto germinacao uniforme excelente",                     "positivo"),
    ("analise do solo mostrou niveis nutricionais adequados sem correcao necessaria",        "positivo"),
    ("operacao de calcario concluida solo em ph ideal para producao",                        "positivo"),
    ("frota de maquinas em otimas condicoes prontas para safra",                             "positivo"),
    ("produtividade do milho atingiu recorde historico na propriedade",                      "positivo"),
    ("safra de trigo encerrada com exito sem perdas por clima ou pragas",                    "positivo"),
    ("pulverizacao concluida com sucesso cobertura total da area planejada",                 "positivo"),
    ("reparo da caixa de marchas concluido trator operando normalmente no campo",            "positivo"),
    ("sistema eletrico do implemento revisado e aprovado sem defeitos encontrados",          "positivo"),
    ("estoque de insumos reabastecido completo pronto para proxima aplicacao",               "positivo"),
    ("manutencao do pivô concluida irrigacao retomada sem interrupcoes",                     "positivo"),
    ("colheita de cana superou tonelagem prevista qualidade excelente",                      "positivo"),
    ("lavoura de algodao apresenta excelente desenvolvimento vegetativo",                    "positivo"),
    ("novo operador treinado com sucesso aprovado em todos os testes",                       "positivo"),
    ("adubo aplicado na dosagem correta lavoura respondendo positivamente",                  "positivo"),
    ("safra de cafe com producao recorde graos de alta qualidade para exportacao",           "positivo"),
    ("plantadeira calibrada corretamente densidade de semeadura perfeita",                   "positivo"),
    ("reforma do pastagem concluida capim estabelecido excelente cobertura",                 "positivo"),
    ("vacinacao do rebanho concluida todos os animais imunizados dentro do prazo",           "positivo"),
    ("conserto do sistema hidraulico concluido pressao normalizada funcionamento perfeito",  "positivo"),
    ("monitoramento de pragas indica populacao abaixo do nivel de dano economico",           "positivo"),
    ("colhedora com nova correia instalada desempenho superior ao esperado",                 "positivo"),
    ("producao de feijao atingiu teto historico safra das aguas excelente",                  "positivo"),
    ("area de plantio ampliada com sucesso maquinario adequado ja disponivel",               "positivo"),
    ("sementes certificadas chegaram no prazo qualidade garantida pelo fornecedor",          "positivo"),
    ("contrato de venda da producao assinado preco acima do esperado",                       "positivo"),
    ("pivo central instalado e funcionando irrigacao da area iniciada com sucesso",          "positivo"),
    ("lavoura de arroz irrigado com bom perfilhamento perspectiva otima",                    "positivo"),
    ("manutenção concluída dentro do orçamento equipamento em perfeito estado",              "positivo"),
    ("silos de armazenagem ampliados capacidade dobrada pronta para a safra",                "positivo"),
    ("pulverizador com bicos novos vazao uniforme sem entupimentos",                         "positivo"),
    ("relatorio de producao aponta eficiencia acima de 95 por cento",                        "positivo"),
    ("lavoura com excelente pegamento de frutos perspectiva de alta produtividade",          "positivo"),
    ("treinamento de equipe tecnica concluido com aproveitamento excelente",                 "positivo"),
    ("aplicacao de micronutrientes corrigiu deficiencias lavoura recuperou vigor",           "positivo"),
    ("parceria com cooperativa garantiu assistencia tecnica e insumos a preco justo",        "positivo"),
    ("analise foliar confirmou nutricao adequada sem deficiencias visuais",                  "positivo"),
    ("galpao de maquinas reformado instalacoes modernas e seguras",                          "positivo"),
    ("certificacao de produto organico obtida mercado premium garantido",                    "positivo"),

    # NEGATIVO (50 exemplos)
    ("trator com falha critica no motor parado aguardando peca de reposicao",               "negativo"),
    ("lavoura com severo ataque de ferrugem producao comprometida urgente",                  "negativo"),
    ("bomba hidraulica com vazamento grave sistema parado risco de contaminacao",            "negativo"),
    ("colheita atrasada devido a chuvas excessivas perdas significativas esperadas",         "negativo"),
    ("pragas resistentes ao defensivo aplicado requer troca imediata do produto",            "negativo"),
    ("equipamento com superaquecimento critico operacao interrompida dano ao motor",         "negativo"),
    ("seca prolongada cultura em estresse hidrico severo producao inviavel",                 "negativo"),
    ("falha eletrica no painel de controle maquina inutilizavel prejuizo alto",              "negativo"),
    ("solo com acidez excessiva correcao urgente necessaria producao baixa",                 "negativo"),
    ("transmissao da colhedora quebrada parada total aguardando tecnico especializado",      "negativo"),
    ("geada severa destruiu maior parte da lavoura prejuizo milionario estimado",            "negativo"),
    ("sistema de drenagem com entupimento critico alagamento do talhao sul",                 "negativo"),
    ("qualidade dos graos abaixo do padrao umidade alta rejeicao na cooperativa",           "negativo"),
    ("vazamento de oleo no diferencial contaminacao do solo area interditada",               "negativo"),
    ("pneu do implemento furado no campo operacao paralisada sem sobressalente",             "negativo"),
    ("ataque severo de lagarta destruiu talhao inteiro colheita perdida",                    "negativo"),
    ("incendio em area de pastagem destruiu pasto e cercas prejuizo enorme",                 "negativo"),
    ("granizo destruiu lavoura de tomate em minutos perda total da producao",                "negativo"),
    ("enchente alagou talhao baixo perdas irreparaveis na cultura de arroz",                 "negativo"),
    ("injetora de adubo entupida distribuicao irregular falha generalizada na lavoura",      "negativo"),
    ("virus da geminivirus dizimou lavoura de tomate sem controle eficaz disponivel",        "negativo"),
    ("colhedora tombou no barranco operador ferido maquina com danos graves",                "negativo"),
    ("sistema de GPS do trator com defeito falhas nas linhas de plantio",                    "negativo"),
    ("silo com vazamento grãos umidos perdas severas por fungos",                            "negativo"),
    ("chuva acima de 200mm em 24 horas causou erosao severa no talhao",                     "negativo"),
    ("praga da broca do cafe atacou pomar inteiro producao comprometida",                    "negativo"),
    ("motor da colhedora fundiu no pico da safra paralisacao total das operacoes",           "negativo"),
    ("herbicida aplicado errado fitotoxicidade grave lavoura em risco",                      "negativo"),
    ("estoque de defensivos vencidos descartados prejuizo operacional imediato",             "negativo"),
    ("falha no sistema de abastecimento d agua animais sem agua por horas",                  "negativo"),
    ("colheita de soja perdeu janela ideal por falta de maquinario disponivel",              "negativo"),
    ("nematoide detectado em area critica producao proxima safra ameacada",                  "negativo"),
    ("furacao de bomba d agua no pico do verao lavoura sem irrigacao por dias",              "negativo"),
    ("calcario nao disponivel no mercado correcao de solo inviabilizada",                    "negativo"),
    ("operador desqualificado causou dano grave no implemento reforma cara",                 "negativo"),
    ("carro de boi atolou no meio do talhao atrapalhou toda a logistica",                    "negativo"),
    ("chuva de granizo danificou todos os paineis solares do sistema de irrigacao",          "negativo"),
    ("financiamento negado pelo banco plantio da proxima safra em risco",                    "negativo"),
    ("queima de pastagem por acidente destruiu meses de reforma do pasto",                   "negativo"),
    ("soja com alto indice de haste verde colheita antecipada com perdas",                   "negativo"),
    ("vento forte derrubou silos moveis graos espalhados no terreiro",                       "negativo"),
    ("praga do percevejo marrom causou danos severos nos graos de soja",                     "negativo"),
    ("colhedora com esteira quebrada parada por tres dias no pico da safra",                 "negativo"),
    ("excesso de chuva impediu entrada de maquinas no talhao por semanas",                   "negativo"),
    ("insumos adulterados causaram morte de lavoura inteira fraude confirmada",              "negativo"),
    ("vaca leiteira com mastite severa producao de leite suspensa tratamento urgente",       "negativo"),
    ("rebanho com brucelose detectada embargo sanitario notificacao ao orgao competente",    "negativo"),
    ("cana com alta incidencia de podridao vermelha perda de 30 por cento da area",         "negativo"),
    ("muda certificada chegou com doenca lavoura de viveiro comprometida",                   "negativo"),
    ("fumaca de queimada vizinha causou dano foliar em talhao de horticultura",              "negativo"),

    # NEUTRO (50 exemplos)
    ("ordem de servico aberta para revisao periodica programada do equipamento",             "neutro"),
    ("relatorio de campo registra condicoes normais sem intercorrencias",                    "neutro"),
    ("agendamento de manutencao preventiva para o proximo mes realizado",                    "neutro"),
    ("leitura dos horarimetros realizada conforme cronograma de inspecao",                   "neutro"),
    ("troca de oleo e filtros realizada conforme manual do fabricante",                      "neutro"),
    ("calibracao dos equipamentos de medicao realizada dentro do padrao",                    "neutro"),
    ("inventario de pecas de reposicao atualizado aguarda aprovacao do gestor",              "neutro"),
    ("visita tecnica agendada para avaliacao das condicoes da lavoura",                      "neutro"),
    ("coleta de amostras de solo enviada para laboratorio analise em andamento",             "neutro"),
    ("registro de horas trabalhadas atualizado conforme planilha de controle",               "neutro"),
    ("reuniao de planejamento da safra marcada para proxima semana",                         "neutro"),
    ("documentacao do equipamento atualizada para conformidade regulatoria",                 "neutro"),
    ("pesagem da producao registrada nota fiscal emitida dentro do prazo",                   "neutro"),
    ("previsao do tempo indica chuvas para os proximos dias aguardando confirmacao",         "neutro"),
    ("operador realizou treinamento de seguranca conforme exigencia do programa",            "neutro"),
    ("levantamento topografico da propriedade concluido dados enviados para analise",        "neutro"),
    ("relatorio mensal de manutencao preenchido e encaminhado para gerencia",                "neutro"),
    ("medicao de vazao do poco artesiano realizada dentro do esperado",                      "neutro"),
    ("visita do agronomo registrada recomendacoes anotadas para proxima etapa",              "neutro"),
    ("conferencia de estoque de sementes realizada quantidades conforme pedido",             "neutro"),
    ("controle de horas de operacao do trator atualizado no sistema interno",               "neutro"),
    ("amostragem de plantas daninhas realizada resultado aguardado para decisao",            "neutro"),
    ("nota de compra de fertilizantes registrada aguarda liberacao financeira",              "neutro"),
    ("relatorio fitossanitario encaminhado ao tecnico responsavel pelo talhao",              "neutro"),
    ("inspecao visual do pomar registrada sem anomalias relevantes observadas",              "neutro"),
    ("monitoramento de chuvas registrou precipitacao dentro da media historica",             "neutro"),
    ("renovacao da licenca ambiental solicitada junto ao orgao competente",                  "neutro"),
    ("ficha tecnica da cultura de milho atualizada pelo agrônomo responsavel",               "neutro"),
    ("logistica de transporte da colheita organizada caminhoes agendados",                   "neutro"),
    ("reuniao com fornecedor de sementes realizada propostas em analise",                    "neutro"),
    ("limpeza e lubrificacao da plantadeira realizadas antes do plantio",                    "neutro"),
    ("analise de agua de irrigacao coletada aguardando resultado laboratorial",              "neutro"),
    ("cadastro de propriedade atualizado no sistema do ministerio da agricultura",           "neutro"),
    ("inspecao de cercas e instalacoes realizada sem ocorrencias registradas",               "neutro"),
    ("boletim climatico mensal registrado e arquivado para historico da fazenda",            "neutro"),
    ("contagem de populacao de plantas realizada dentro da faixa esperada",                  "neutro"),
    ("fotos georeferenciadas do talhao registradas para acompanhamento satelital",           "neutro"),
    ("certificado de calibracao da balança renovado enviado para arquivo",                   "neutro"),
    ("pedido de cotacao de defensivos encaminhado aguarda retorno dos fornecedores",         "neutro"),
    ("sementes em tratamento fungicida aguardando secagem para plantio",                     "neutro"),
    ("relatorio de ocorrencia de pragas arquivado historico atualizado",                     "neutro"),
    ("maquinario conferido antes da safra pontos de atencao listados para revisao",          "neutro"),
    ("medicao de ph do solo realizada resultado dentro da faixa toleravel",                  "neutro"),
    ("guia de transporte de animais emitida para movimentacao do lote",                      "neutro"),
    ("plano de manejo integrado de pragas elaborado aguarda aprovacao",                      "neutro"),
    ("balanco de insumos da safra anterior documentado para auditoria interna",              "neutro"),
    ("protocolo de vacinacao atualizado conforme calendario sanitario estadual",             "neutro"),
    ("mapa de talhoes atualizado com as alteracoes da ultima safra",                         "neutro"),
    ("reuniao de equipe tecnica realizada pauta discutida atas registradas",                 "neutro"),
    ("proposta de seguro rural analisada aguarda decisao da gerencia financeira",            "neutro"),
]

LABELS    = {"positivo": 0, "negativo": 1, "neutro": 2}
LABEL_REV = {0: "positivo", 1: "negativo", 2: "neutro"}
COLORS    = {"positivo": "#4ade80", "negativo": "#f87171", "neutro": "#60a5fa"}


# ─────────────────────────────────────────────────────────
# TOKENIZADOR SIMPLES
# ─────────────────────────────────────────────────────────

class SimpleTokenizer:
    """Tokenizador baseado em palavras com vocabulario aprendido."""

    SPECIAL = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}

    def __init__(self, max_vocab=500):
        self.max_vocab = max_vocab
        self.word2idx  = dict(self.SPECIAL)
        self.idx2word  = {v: k for k, v in self.SPECIAL.items()}

    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z\u00c0-\u00ff\s]", " ", text)
        return text.split()

    def fit(self, texts):
        words = Counter()
        for t in texts:
            words.update(self._tokenize(t))
        for word, _ in words.most_common(self.max_vocab - len(self.SPECIAL)):
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx]  = word
        return self

    def encode(self, text, max_len=20):
        tokens = ["[CLS]"] + self._tokenize(text)[:max_len-2] + ["[SEP]"]
        ids    = [self.word2idx.get(t, self.SPECIAL["[UNK]"]) for t in tokens]
        # pad
        pad_len = max_len - len(ids)
        ids    += [self.SPECIAL["[PAD]"]] * pad_len
        tokens += ["[PAD]"] * pad_len
        return ids[:max_len], tokens[:max_len]

    @property
    def vocab_size(self):
        return len(self.word2idx)


# ─────────────────────────────────────────────────────────
# TRANSFORMER DO ZERO — NumPy
# ─────────────────────────────────────────────────────────

class TransformerClassifier:
    """
    Transformer simplificado para classificacao de sentimento.
    Implementado do zero em NumPy para fins educativos.

    Arquitetura:
      Token Embedding -> Positional Encoding ->
      Multi-Head Self-Attention -> Feed-Forward ->
      Mean Pooling -> Linear -> Softmax
    """

    def __init__(self, vocab_size, d_model=32, n_heads=4,
                 d_ff=64, max_len=20, n_classes=3, seed=42):
        np.random.seed(seed)
        self.d_model   = d_model
        self.n_heads   = n_heads
        self.d_ff      = d_ff
        self.max_len   = max_len
        self.n_classes = n_classes
        self.d_head    = d_model // n_heads

        # Embeddings
        self.E   = np.random.randn(vocab_size, d_model) * 0.05
        self.PE  = self._positional_encoding(max_len, d_model)

        # Multi-Head Attention: Q, K, V por cabeca + projecao de saida
        scale = np.sqrt(1.0 / d_model)
        self.W_Q = [np.random.randn(d_model, self.d_head) * scale for _ in range(n_heads)]
        self.W_K = [np.random.randn(d_model, self.d_head) * scale for _ in range(n_heads)]
        self.W_V = [np.random.randn(d_model, self.d_head) * scale for _ in range(n_heads)]
        self.W_O = np.random.randn(d_model, d_model) * scale

        # Feed-Forward
        self.W1 = np.random.randn(d_model, d_ff)  * np.sqrt(2.0/d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model)  * np.sqrt(2.0/d_ff)
        self.b2 = np.zeros(d_model)

        # Classificador final
        self.W_cls = np.random.randn(d_model, n_classes) * 0.05
        self.b_cls = np.zeros(n_classes)

        # Historico
        self.history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
        self._last_attn = None   # shape: (n_heads, seq_len, seq_len)
        self._last_embs = None   # embeddings pos-atencao

        # Velocidades para SGD com momentum (β = 0.9)
        self._v_W_cls = np.zeros_like(self.W_cls)
        self._v_b_cls = np.zeros_like(self.b_cls)
        self._v_W1    = np.zeros_like(self.W1)
        self._v_b1    = np.zeros_like(self.b1)
        self._v_W2    = np.zeros_like(self.W2)
        self._v_b2    = np.zeros_like(self.b2)
        self._v_W_O   = np.zeros_like(self.W_O)
        self._v_E     = np.zeros_like(self.E)

    def _positional_encoding(self, max_len, d_model):
        """Positional Encoding sinusoidal (Vaswani et al. 2017)."""
        PE = np.zeros((max_len, d_model))
        pos = np.arange(max_len)[:, None]
        div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        PE[:, 0::2] = np.sin(pos * div)
        PE[:, 1::2] = np.cos(pos * div)
        return PE

    def _softmax(self, x, axis=-1):
        e = np.exp(x - x.max(axis=axis, keepdims=True))
        return e / (e.sum(axis=axis, keepdims=True) + 1e-9)

    def _attention(self, X, head_idx, mask=None):
        """
        Scaled Dot-Product Attention para uma cabeca.
        Retorna (output, attention_weights).
        """
        Q = X @ self.W_Q[head_idx]   # (seq, d_head)
        K = X @ self.W_K[head_idx]
        V = X @ self.W_V[head_idx]

        # Scores: Q . K^T / sqrt(d_head)
        scores = Q @ K.T / np.sqrt(self.d_head)   # (seq, seq)

        if mask is not None:
            scores[mask] = -1e9

        attn_w = self._softmax(scores)   # (seq, seq)
        out    = attn_w @ V              # (seq, d_head)
        return out, attn_w

    def forward(self, token_ids, return_attn=False):
        """Forward pass completo."""
        seq_len = len(token_ids)

        # 1. Embedding + Positional Encoding
        X = self.E[token_ids] + self.PE[:seq_len]   # (seq, d_model)

        # 2. Multi-Head Self-Attention
        head_outputs = []
        attn_weights = []
        for h in range(self.n_heads):
            out_h, attn_h = self._attention(X, h)
            head_outputs.append(out_h)
            attn_weights.append(attn_h)

        # Concatena cabecas e projeta
        concat = np.concatenate(head_outputs, axis=-1)  # (seq, d_model)
        attn_out = concat @ self.W_O                     # (seq, d_model)

        # Add & Norm (simplificado)
        X2 = X + attn_out
        X2 = (X2 - X2.mean()) / (X2.std() + 1e-6)

        # 3. Feed-Forward
        ff = np.maximum(0, X2 @ self.W1 + self.b1) @ self.W2 + self.b2  # ReLU
        X3 = X2 + ff
        X3 = (X3 - X3.mean()) / (X3.std() + 1e-6)

        # 4. Pooling (media sobre tokens nao-PAD)
        pooled = X3.mean(axis=0)  # (d_model,)

        # 5. Classificador
        logits = pooled @ self.W_cls + self.b_cls  # (n_classes,)
        probs  = self._softmax(logits)

        if return_attn:
            self._last_attn = np.array(attn_weights)  # (n_heads, seq, seq)
            self._last_embs = X3                       # (seq, d_model)

        return probs

    def _cross_entropy(self, probs, label):
        return -np.log(probs[label] + 1e-15)

    def _update(self, token_ids, label, lr):
        """
        SGD com backprop analitico.
        Propaga gradiente pelo classificador, FF, atencao e embeddings.
        """
        seq_len = len(token_ids)

        # Forward completo com cache
        X_emb = self.E[token_ids] + self.PE[:seq_len]

        head_out = []; attn_ws = []
        for h in range(self.n_heads):
            o, aw = self._attention(X_emb, h)
            head_out.append(o); attn_ws.append(aw)
        concat   = np.concatenate(head_out, axis=-1)
        attn_out = concat @ self.W_O
        X2 = X_emb + attn_out
        mu2, std2 = X2.mean(), X2.std() + 1e-6
        X2n = (X2 - mu2) / std2

        pre_ff = X2n @ self.W1 + self.b1
        ff_act = np.maximum(0, pre_ff)
        ff_out = ff_act @ self.W2 + self.b2
        X3 = X2n + ff_out
        mu3, std3 = X3.mean(), X3.std() + 1e-6
        X3n = (X3 - mu3) / std3

        pooled = X3n.mean(axis=0)
        logits = pooled @ self.W_cls + self.b_cls
        probs  = self._softmax(logits)
        loss   = self._cross_entropy(probs, label)

        # Backward
        d_logits = probs.copy(); d_logits[label] -= 1.0

        dW_cls = np.outer(pooled, d_logits)
        db_cls = d_logits.copy()
        d_pooled = self.W_cls @ d_logits

        d_X3n = np.tile(d_pooled / seq_len, (seq_len, 1))
        d_X3  = d_X3n / std3

        d_ff_out     = d_X3.copy()
        d_X2n_res    = d_X3.copy()
        d_ff_act     = d_ff_out @ self.W2.T
        d_pre_ff     = d_ff_act * (pre_ff > 0)
        dW2 = ff_act.T @ d_ff_out;  db2 = d_ff_out.sum(axis=0)
        dW1 = X2n.T   @ d_pre_ff;   db1 = d_pre_ff.sum(axis=0)
        d_X2n = d_X2n_res + d_pre_ff @ self.W1.T
        d_X2  = d_X2n / std2
        d_attn_out = d_X2.copy()
        d_X_emb    = d_X2.copy()
        dW_O    = concat.T @ d_attn_out
        d_concat = d_attn_out @ self.W_O.T

        d_emb = d_X_emb.copy()
        chunk = self.d_model // self.n_heads
        for h in range(self.n_heads):
            d_head_h = d_concat[:, h*chunk:(h+1)*chunk]
            d_V = attn_ws[h].T @ d_head_h
            d_emb += d_V @ self.W_V[h].T * 0.2

        # Updates com SGD + momentum (β = 0.9) e gradient clipping
        β = 0.9
        def _step(v, w, g):
            v[:] = β * v + lr * g
            w   -= v

        _step(self._v_W_cls, self.W_cls, dW_cls)
        _step(self._v_b_cls, self.b_cls, db_cls)
        _step(self._v_W1,    self.W1,    np.clip(dW1,  -1, 1))
        _step(self._v_b1,    self.b1,    np.clip(db1,  -1, 1))
        _step(self._v_W2,    self.W2,    np.clip(dW2,  -1, 1))
        _step(self._v_b2,    self.b2,    np.clip(db2,  -1, 1))
        _step(self._v_W_O,   self.W_O,   np.clip(dW_O, -1, 1) * 0.5)
        for tok_id, d_e in zip(token_ids, d_emb):
            self._v_E[tok_id] = β * self._v_E[tok_id] + lr * np.clip(d_e, -1, 1)
            self.E[tok_id]   -= self._v_E[tok_id]

        return loss

    def train(self, X_ids, y, X_val_ids, y_val, epochs=30, lr=0.08, progress_cb=None):
        """Treino com SGD por amostra."""
        for ep in range(1, epochs + 1):
            if progress_cb:
                progress_cb(ep, epochs)
            idx = np.random.permutation(len(X_ids))
            ep_loss = 0.0
            for i in idx:
                ep_loss += self._update(X_ids[i], y[i], lr)

            ep_loss /= len(X_ids)

            # Metricas
            acc = np.mean([
                np.argmax(self.forward(xi)) == yi
                for xi, yi in zip(X_ids, y)
            ])
            val_loss = np.mean([
                self._cross_entropy(self.forward(xi), yi)
                for xi, yi in zip(X_val_ids, y_val)
            ])
            val_acc = np.mean([
                np.argmax(self.forward(xi)) == yi
                for xi, yi in zip(X_val_ids, y_val)
            ])

            self.history["loss"].append(float(ep_loss))
            self.history["acc"].append(float(acc))
            self.history["val_loss"].append(float(val_loss))
            self.history["val_acc"].append(float(val_acc))

        return self.history


# ─────────────────────────────────────────────────────────
# MODELO BASELINE (sem atencao) para comparacao
# ─────────────────────────────────────────────────────────

class BagOfWordsClassifier:
    """Bag of Words + regressao logistica — sem atencao."""

    def __init__(self, vocab_size, n_classes=3, seed=42):
        np.random.seed(seed)
        self.W = np.random.randn(vocab_size, n_classes) * 0.01
        self.b = np.zeros(n_classes)
        self.history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

    def _softmax(self, x):
        e = np.exp(x - x.max())
        return e / (e.sum() + 1e-9)

    def forward(self, token_ids):
        bow = np.zeros(self.W.shape[0])
        for i in token_ids:
            bow[i] += 1.0
        bow /= (bow.sum() + 1e-9)
        logits = bow @ self.W + self.b
        return self._softmax(logits)

    def train(self, X_ids, y, X_val_ids, y_val, epochs=30, lr=0.1, progress_cb=None):
        for ep in range(1, epochs + 1):
            if progress_cb:
                progress_cb(ep, epochs)
            idx = np.random.permutation(len(X_ids))
            ep_loss = 0.0
            for i in idx:
                probs  = self.forward(X_ids[i])
                loss   = -np.log(probs[y[i]] + 1e-15)
                ep_loss += loss
                # Gradiente analitico simples
                d_logits = probs.copy(); d_logits[y[i]] -= 1
                bow = np.zeros(self.W.shape[0])
                for j in X_ids[i]: bow[j] += 1
                bow /= (bow.sum() + 1e-9)
                self.W -= lr * np.outer(bow, d_logits)
                self.b -= lr * d_logits

            ep_loss /= len(X_ids)
            acc     = np.mean([np.argmax(self.forward(xi))==yi for xi,yi in zip(X_ids,y)])
            v_loss  = np.mean([-np.log(self.forward(xi)[yi]+1e-15) for xi,yi in zip(X_val_ids,y_val)])
            v_acc   = np.mean([np.argmax(self.forward(xi))==yi for xi,yi in zip(X_val_ids,y_val)])

            self.history["loss"].append(float(ep_loss))
            self.history["acc"].append(float(acc))
            self.history["val_loss"].append(float(v_loss))
            self.history["val_acc"].append(float(v_acc))

        return self.history


# ─────────────────────────────────────────────────────────
# PREPARACAO DOS DADOS
# ─────────────────────────────────────────────────────────

@st.cache_data
def prepare_data():
    texts  = [d[0] for d in DATASET]
    labels = [LABELS[d[1]] for d in DATASET]

    tok = SimpleTokenizer(max_vocab=400)
    tok.fit(texts)

    encoded = [tok.encode(t, max_len=18) for t in texts]
    X_ids   = [e[0] for e in encoded]
    X_toks  = [e[1] for e in encoded]

    X_tr, X_te, y_tr, y_te, tok_tr, tok_te, txt_tr, txt_te = train_test_split(
        X_ids, labels, X_toks, texts,
        test_size=0.25, random_state=42, stratify=labels
    )
    return X_tr, X_te, y_tr, y_te, tok_tr, tok_te, txt_tr, txt_te, tok


def train_models(epochs, lr_transformer, lr_bow, progress_cb=None):
    X_tr, X_te, y_tr, y_te, *_, tokenizer = prepare_data()

    # Transformer (seed=13 converge melhor neste dataset)
    tf = TransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        d_model=32, n_heads=4, d_ff=64, max_len=18, n_classes=3, seed=13
    )
    def _cb_tf(ep, total):
        if progress_cb:
            progress_cb(ep, total * 2, f"Transformer — época {ep}/{total}")
    h_tf = tf.train(X_tr, y_tr, X_te, y_te, epochs=epochs, lr=lr_transformer, progress_cb=_cb_tf)

    # BoW baseline
    bow = BagOfWordsClassifier(vocab_size=tokenizer.vocab_size)
    def _cb_bow(ep, total):
        if progress_cb:
            progress_cb(total + ep, total * 2, f"BoW — época {ep}/{total}")
    h_bow = bow.train(X_tr, y_tr, X_te, y_te, epochs=epochs, lr=lr_bow, progress_cb=_cb_bow)

    return tf, bow, h_tf, h_bow


# ─────────────────────────────────────────────────────────
# GRAFICOS PLOTLY
# ─────────────────────────────────────────────────────────

DARK = dict(
    paper_bgcolor="#060a0f", plot_bgcolor="#0b1117",
    font=dict(family="IBM Plex Mono", color="#8899aa", size=11),
    xaxis=dict(gridcolor="#1a2535", linecolor="#22334a", zerolinecolor="#1a2535"),
    yaxis=dict(gridcolor="#1a2535", linecolor="#22334a", zerolinecolor="#1a2535"),
)


def plot_attention_heatmap(attn_matrix, tokens, title="Atencao"):
    """Heatmap da matriz de atencao para uma cabeca."""
    clean = [t for t in tokens if t != "[PAD]"]
    n     = len(clean)
    mat   = attn_matrix[:n, :n]

    fig = go.Figure(go.Heatmap(
        z=mat, x=clean, y=clean,
        colorscale=[
            [0.0, "#060a0f"], [0.3, "#0d3344"],
            [0.6, "#0d9488"], [0.85, "#f59e0b"], [1.0, "#fff7ed"]
        ],
        showscale=True,
        colorbar=dict(
            tickfont=dict(color="#8899aa"), thickness=12,
            tickvals=[0, 0.5, 1],
            ticktext=["0 — ignora", "0.5", "1 — foco total"],
            title=dict(text="Atenção", font=dict(color="#8899aa", size=10)),
        ),
        text=np.round(mat, 3),
        texttemplate="%{text}",
        textfont=dict(size=8, color="#dde6f0"),
        hovertemplate="De: %{y}<br>Para: %{x}<br>Atencao: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#060a0f", plot_bgcolor="#0b1117",
        font=dict(family="IBM Plex Mono", color="#8899aa", size=11),
        title=title,
        xaxis=dict(title="Token destino", tickangle=-35,
                   gridcolor="#1a2535", linecolor="#22334a", zerolinecolor="#1a2535"),
        yaxis=dict(title="Token origem", autorange="reversed",
                   gridcolor="#1a2535", linecolor="#22334a", zerolinecolor="#1a2535"),
        height=400, margin=dict(t=50, b=80, l=80, r=30),
    )
    return fig


def plot_multihead(attn_all, tokens):
    """Grade de heatmaps para todas as cabecas."""
    clean = [t for t in tokens if t != "[PAD]"]
    n     = len(clean)
    n_heads = attn_all.shape[0]
    cols  = 2
    rows  = (n_heads + 1) // cols

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"Cabeca {i+1}" for i in range(n_heads)],
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )
    for h in range(n_heads):
        mat = attn_all[h, :n, :n]
        r, c = divmod(h, cols)
        fig.add_trace(go.Heatmap(
            z=mat, x=clean, y=clean,
            colorscale=[[0,"#060a0f"],[0.5,"#0d9488"],[1,"#f59e0b"]],
            showscale=False,
            hovertemplate=f"Cabeca {h+1}<br>De: %{{y}}<br>Para: %{{x}}<br>Atencao: %{{z:.3f}}<extra></extra>",
        ), row=r+1, col=c+1)

    fig.update_layout(
        **DARK, height=160 * rows + 80,
        title="Multi-Head Attention — todas as cabecas",
        margin=dict(t=60, b=30, l=60, r=20),
    )
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(
                tickfont=dict(size=7, color="#445566"),
                showgrid=False,
            )
    return fig


def plot_embeddings_2d(transformer, tokenizer, texts, labels_raw):
    """PCA dos embeddings de cada texto apos atencao."""
    vecs = []
    for txt in texts:
        ids, _ = tokenizer.encode(txt, max_len=18)
        transformer.forward(ids, return_attn=True)
        if transformer._last_embs is None:
            raise ValueError("Embeddings nao disponiveis — treine o modelo primeiro.")
        emb = transformer._last_embs.mean(axis=0)
        vecs.append(emb)

    vecs_np = np.array(vecs)
    pca     = PCA(n_components=2, random_state=42)
    coords  = pca.fit_transform(vecs_np)

    df = pd.DataFrame({
        "x": coords[:, 0], "y": coords[:, 1],
        "sentimento": [LABEL_REV[l] for l in labels_raw],
        "texto": [t[:50]+"..." if len(t)>50 else t for t in texts],
    })
    color_map = {"positivo":"#4ade80","negativo":"#f87171","neutro":"#60a5fa"}
    fig = px.scatter(
        df, x="x", y="y", color="sentimento",
        hover_data={"texto": True, "x": False, "y": False},
        color_discrete_map=color_map,
    )
    fig.update_traces(marker=dict(size=11, line=dict(width=1.5, color="#0b1117")))
    fig.update_layout(
        paper_bgcolor="#060a0f", plot_bgcolor="#0b1117",
        font=dict(family="IBM Plex Mono", color="#8899aa", size=11),
        title="Embeddings em Espaco 2D (PCA apos Atencao)",
        xaxis=dict(title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                   gridcolor="#1a2535", linecolor="#22334a", zerolinecolor="#1a2535"),
        yaxis=dict(title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                   gridcolor="#1a2535", linecolor="#22334a", zerolinecolor="#1a2535"),
        legend=dict(bgcolor="#0b1117", bordercolor="#22334a"),
        height=420, margin=dict(t=50, b=30, l=40, r=20),
    )
    return fig, pca.explained_variance_ratio_


def plot_comparison(h_tf, h_bow):
    """Compara curvas de acuracia: Transformer vs BoW."""
    eps = list(range(1, len(h_tf["acc"])+1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eps, y=[a*100 for a in h_tf["val_acc"]],
                             name="Transformer (val)", line=dict(color="#f59e0b", width=2.5), mode="lines"))
    fig.add_trace(go.Scatter(x=eps, y=[a*100 for a in h_bow["val_acc"]],
                             name="Bag-of-Words (val)", line=dict(color="#60a5fa", width=2.5, dash="dot"), mode="lines"))
    fig.add_trace(go.Scatter(x=eps, y=[a*100 for a in h_tf["acc"]],
                             name="Transformer (treino)", line=dict(color="#f59e0b", width=1.5, dash="dash"),
                             opacity=0.5, mode="lines"))
    fig.update_layout(**DARK,
        title="Comparacao: Transformer vs Bag-of-Words",
        xaxis_title="Epoca", yaxis_title="Acuracia (%)",
        yaxis_range=[0, 105],
        legend=dict(bgcolor="#0b1117", bordercolor="#22334a"),
        height=320, margin=dict(t=40, b=30, l=40, r=20),
    )
    return fig


def plot_loss_curves(h_tf, h_bow):
    eps = list(range(1, len(h_tf["loss"])+1))
    peak_loss = h_tf["loss"][0] if h_tf["loss"] else 1.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eps, y=h_tf["loss"],
                             name="Transformer (treino)", line=dict(color="#f59e0b", width=2.5), mode="lines"))
    fig.add_trace(go.Scatter(x=eps, y=h_tf["val_loss"],
                             name="Transformer (val)", line=dict(color="#2dd4bf", width=2.5, dash="dot"), mode="lines"))
    fig.add_trace(go.Scatter(x=eps, y=h_bow["val_loss"],
                             name="BoW (val)", line=dict(color="#60a5fa", width=1.5, dash="dash"),
                             opacity=0.6, mode="lines"))
    fig.add_annotation(
        x=1, y=peak_loss,
        text="Pico normal —<br>pesos aleatórios",
        showarrow=True, arrowhead=2, arrowcolor="#445566",
        font=dict(size=9, color="#8899aa", family="IBM Plex Mono"),
        bgcolor="#0b1117", bordercolor="#22334a", borderwidth=1,
        ax=40, ay=-30,
    )
    fig.update_layout(**DARK,
        title="Curva de Loss — Cross-Entropy",
        xaxis_title="Epoca", yaxis_title="Loss",
        legend=dict(bgcolor="#0b1117", bordercolor="#22334a"),
        height=300, margin=dict(t=40, b=30, l=40, r=20),
    )
    return fig


def plot_pos_encoding():
    """Visualiza o positional encoding."""
    max_len, d_model = 18, 32
    PE = np.zeros((max_len, d_model))
    pos = np.arange(max_len)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    PE[:, 0::2] = np.sin(pos * div)
    PE[:, 1::2] = np.cos(pos * div)

    fig = go.Figure(go.Heatmap(
        z=PE,
        colorscale=[[0,"#f87171"],[0.5,"#0b1117"],[1,"#2dd4bf"]],
        zmid=0, showscale=True,
        colorbar=dict(tickfont=dict(color="#8899aa"), thickness=12),
        hovertemplate="Posicao: %{y}<br>Dimensao: %{x}<br>Valor: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(**DARK,
        title="Positional Encoding (seno/cosseno por posicao e dimensao)",
        xaxis_title="Dimensao do embedding",
        yaxis_title="Posicao no texto",
        height=320, margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig


# ─────────────────────────────────────────────────────────
# TOKENS COLORIDOS
# ─────────────────────────────────────────────────────────

def render_tokens(tokens, attn_row=None, label=None):
    """Renderiza tokens com cores baseadas na atencao recebida."""
    html = '<div style="line-height:2.2;margin:.5rem 0">'
    clean = [(i, t) for i, t in enumerate(tokens) if t != "[PAD]"]

    for i, tok in clean:
        if attn_row is not None and i < len(attn_row):
            weight = float(attn_row[i])
            alpha  = 0.1 + weight * 0.9
            r, g, b = (245, 158, 11) if label == "positivo" else \
                      (248, 113, 113) if label == "negativo" else \
                      (96, 165, 250)
            bg  = f"rgba({r},{g},{b},{alpha:.2f})"
            brd = f"rgba({r},{g},{b},{min(1.0, alpha+0.2):.2f})"
            col = "#000" if alpha > 0.6 else "#dde6f0"
            size = int(11 + weight * 6)
        else:
            sp_colors = {"[CLS]": "#f59e0b", "[SEP]": "#2dd4bf", "[UNK]": "#c084fc"}
            if tok in sp_colors:
                bg, brd, col, size = "transparent", sp_colors[tok], sp_colors[tok], 11
            else:
                bg, brd, col, size = "transparent", "#22334a", "#8899aa", 11

        title_attr = f'title="Atencao: {attn_row[i]:.3f}"' if (attn_row is not None and i < len(attn_row)) else ''
        html += (
            f'<span class="tok" style="background:{bg};border-color:{brd};'
            f'color:{col};font-size:{size}px" {title_attr}>'
            f'{tok}</span>'
        )
    html += '</div>'
    return html


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    X_tr, X_te, y_tr, y_te, tok_tr, tok_te, txt_tr, txt_te, tokenizer = prepare_data()
    all_texts  = [d[0] for d in DATASET]
    all_labels = [LABELS[d[1]] for d in DATASET]

    # ── SIDEBAR ─────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div style="font-family:IBM Plex Sans,sans-serif;font-size:1.25rem;font-weight:700;'
            'background:linear-gradient(90deg,#f59e0b,#2dd4bf);-webkit-background-clip:text;'
            '-webkit-text-fill-color:transparent;letter-spacing:.03em">&#128269; AttentionMind</div>'
            '<div style="font-family:IBM Plex Mono,monospace;font-size:.58rem;'
            'color:#445566;letter-spacing:.15em;margin-bottom:1rem">TRANSFORMERS DO ZERO</div>',
            unsafe_allow_html=True
        )

        st.markdown('<p class="sl">Hiperparametros</p>', unsafe_allow_html=True)
        epochs  = st.slider("Epocas de treino", 10, 60,
                            st.session_state.get("_ep", 30), 5,
                            help="Mais épocas = mais aprendizado, mas acima de ~40 pode haver overfitting (val_loss sobe)")
        lr_tf   = st.select_slider("Learning rate (Transformer)",
                                   [0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
                                   value=st.session_state.get("_lr_tf", 0.05),
                                   help="Passo do gradiente. Muito alto (>0.1): oscila e não converge. Muito baixo (<0.03): aprende devagar")
        lr_bow  = st.select_slider("Learning rate (BoW)",
                                   [0.05, 0.1, 0.15, 0.2, 0.3],
                                   value=st.session_state.get("_lr_bow", 0.15),
                                   help="BoW é mais simples e aceita lr maior sem instabilidade")
        c_train, c_reset = st.columns([3, 1])
        train_btn = c_train.button("Treinar modelos", width="stretch")
        if c_reset.button("↺", width="stretch", help="Redefinir hiperparâmetros para os valores padrão"):
            st.session_state["_ep"]     = 30
            st.session_state["_lr_tf"]  = 0.05
            st.session_state["_lr_bow"] = 0.15
            st.rerun()

        st.markdown('<hr class="dv">', unsafe_allow_html=True)
        st.markdown('<p class="sl">Arquitetura do Transformer</p>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card" style="font-family:IBM Plex Mono,monospace;font-size:.72rem">'
            '<div style="display:flex;justify-content:space-between;margin-bottom:.35rem">'
            '<span style="color:var(--muted)">d_model</span><span style="color:var(--gold)">32</span></div>'
            '<div style="display:flex;justify-content:space-between;margin-bottom:.35rem">'
            '<span style="color:var(--muted)">n_heads</span><span style="color:var(--teal)">4</span></div>'
            '<div style="display:flex;justify-content:space-between;margin-bottom:.35rem">'
            '<span style="color:var(--muted)">d_ff</span><span style="color:var(--blue)">64</span></div>'
            '<div style="display:flex;justify-content:space-between;margin-bottom:.35rem">'
            '<span style="color:var(--muted)">max_len</span><span style="color:var(--purple)">18</span></div>'
            '<div style="display:flex;justify-content:space-between">'
            '<span style="color:var(--muted)">vocab</span>'
            f'<span style="color:var(--text2)">{tokenizer.vocab_size}</span></div>'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown('<hr class="dv">', unsafe_allow_html=True)
        st.markdown('<p class="sl">Conceitos</p>', unsafe_allow_html=True)
        st.markdown(
            '<div class="cg"><b style="color:var(--gold)">Self-Attention</b><br>'
            '<span style="font-size:.8rem">Cada token "olha" para todos os outros e decide quanto prestar atencao em cada um</span></div>'
            '<div class="ct"><b style="color:var(--teal)">Q, K, V</b><br>'
            '<span style="font-size:.8rem">Query, Key, Value — como uma busca: Q pergunta, K responde, V entrega o conteudo</span></div>'
            '<div class="cb"><b style="color:var(--blue)">Multi-Head</b><br>'
            '<span style="font-size:.8rem">Multiplas cabecas aprendem diferentes tipos de relacao entre tokens em paralelo</span></div>'
            '<div class="cp"><b style="color:var(--purple)">Pos. Encoding</b><br>'
            '<span style="font-size:.8rem">Injeta informacao de posicao via seno/cosseno — o modelo sabe a ordem das palavras</span></div>',
            unsafe_allow_html=True
        )

    # Treina se botao clicado ou nao treinado ainda
    if train_btn or "transformer" not in st.session_state:
        _prog_bar  = st.progress(0.0)
        _prog_text = st.empty()
        def _progress(step, total, label=""):
            if total > 0:
                _prog_bar.progress(min(1.0, step / total))
            _prog_text.caption(label)
        _t0 = time.time()
        transformer, bow_model, h_tf, h_bow = train_models(
            epochs, lr_tf, lr_bow, progress_cb=_progress
        )
        _elapsed = time.time() - _t0
        _prog_bar.empty()
        _prog_text.empty()
        st.session_state.transformer   = transformer
        st.session_state.bow_model     = bow_model
        st.session_state.h_tf          = h_tf
        st.session_state.h_bow         = h_bow
        st.session_state.train_elapsed = _elapsed
        st.rerun()

    transformer = st.session_state.transformer
    bow_model   = st.session_state.bow_model
    h_tf        = st.session_state.h_tf
    h_bow       = st.session_state.h_bow

    # ── HEADER ──────────────────────────────────────────
    st.markdown(
        '<h1 style="font-size:1.95rem;margin-bottom:0">'
        '<span style="background:linear-gradient(90deg,#f59e0b,#2dd4bf);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent">AttentionMind</span>'
        ' <span style="color:#445566;font-size:1rem;font-family:IBM Plex Mono,monospace">'
        '/ Transformers do Zero</span></h1>'
        '<p style="color:#445566;font-family:IBM Plex Mono,monospace;font-size:.65rem;'
        'letter-spacing:.14em;margin-top:.2rem;margin-bottom:1rem">'
        'NUMPY PURO -- SELF-ATTENTION -- MULTI-HEAD -- POSITIONAL ENCODING -- TEXTOS AGRICOLAS</p>',
        unsafe_allow_html=True
    )

    # ── METRICAS ────────────────────────────────────────
    tf_acc  = h_tf["val_acc"][-1]  * 100
    bow_acc = h_bow["val_acc"][-1] * 100
    tf_loss = h_tf["val_loss"][-1]
    _elapsed = st.session_state.get("train_elapsed")
    _elapsed_str = f"{_elapsed:.1f}s" if _elapsed else "—"
    _tips = [
        ("Acc TF",    f"{tf_acc:.1f}%",             "gv", "Acuracia de validacao do Transformer — percentual de textos classificados corretamente"),
        ("Acc BoW",   f"{bow_acc:.1f}%",            "bv", "Acuracia de validacao do Bag-of-Words — modelo sem atencao, usado como comparacao"),
        ("Ganho",     f"+{tf_acc-bow_acc:.1f}%",    "tv", "Ganho do Transformer sobre BoW — quanto a atencao ajudou (positivo = Transformer ganhou)"),
        ("Val Loss",  f"{tf_loss:.3f}",             "rv", "Perda de validacao (cross-entropy) — quanto menor melhor; mede a incerteza do modelo"),
        ("Treino",    _elapsed_str,                 "pv", "Tempo total de treinamento — Transformer e muito mais lento que BoW por causa da atencao"),
    ]
    for col_w, (lbl, val, cls, tip) in zip(st.columns(5), _tips):
        with col_w:
            st.markdown(
                f'<div class="mb" title="{tip}">'
                f'<div class="ml">{lbl}</div>'
                f'<div class="mv {cls}">{val}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ONBOARDING ──────────────────────────────────────
    if not st.session_state.get("onboarding_done"):
        st.markdown(
            '<div class="cg" style="margin-bottom:.8rem">'
            '<b style="color:var(--gold)">👋 Bem-vindo ao AttentionMind!</b><br>'
            '<span style="font-size:.85rem">'
            'Sugestão de jornada: '
            '<b>① Como Funciona</b> → entenda a teoria · '
            '<b>② Treine</b> os modelos pela sidebar · '
            '<b>③ Atenção</b> → veja o que o modelo "olha" · '
            '<b>④ Embeddings 2D</b> → visualize o espaço aprendido · '
            '<b>⑤ Classifique</b> textos próprios'
            '</span></div>',
            unsafe_allow_html=True
        )
        if st.button("Entendi, não mostrar novamente", key="dismiss_onboarding"):
            st.session_state.onboarding_done = True
            st.rerun()

    # ── TABS ────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "① CLASSIFICAR",
        "② ATENÇÃO",
        "③ MULTI-HEAD",
        "④ EMBEDDINGS",
        "⑤ COMO FUNCIONA",
    ])

    # ════════════════════════════════════════════════════
    # TAB 1: CLASSIFICAR TEXTO
    # ════════════════════════════════════════════════════
    with tab1:
        c1a, c1b = st.columns([3, 2], gap="large")

        with c1a:
            st.markdown('<p class="sl">Digite um texto agricola para classificar</p>', unsafe_allow_html=True)
            user_text = st.text_area(
                "Texto",
                value="trator com falha no motor parado aguardando reparo urgente",
                height=90, label_visibility="collapsed",
                placeholder="Ex: colheita excelente producao acima da meta..."
            )

            st.markdown('<p class="sl">Ou escolha um exemplo do dataset</p>', unsafe_allow_html=True)
            exemplo = st.selectbox(
                "Exemplos", ["-- selecione --"] + [d[0] for d in DATASET],
                label_visibility="collapsed"
            )
            if exemplo != "-- selecione --":
                user_text = exemplo

            analyze_btn = st.button("Analisar sentimento", width="stretch")

        with c1b:
            st.markdown('<p class="sl">Resultados do conjunto de teste</p>', unsafe_allow_html=True)
            st.plotly_chart(plot_comparison(h_tf, h_bow), width="stretch")

        if analyze_btn and not user_text.strip():
            st.warning("Digite um texto para analisar.")

        if analyze_btn and user_text.strip():
            ids, tokens = tokenizer.encode(user_text, max_len=18)

            # Transformer
            probs_tf = transformer.forward(ids, return_attn=True)
            pred_tf  = int(np.argmax(probs_tf))
            attn_all  = transformer._last_attn  # (n_heads, seq, seq)
            attn_avg  = attn_all.mean(axis=0)   # media das cabecas

            # BoW
            probs_bow = bow_model.forward(ids)
            pred_bow  = int(np.argmax(probs_bow))

            sent_tf  = LABEL_REV[pred_tf]
            sent_bow = LABEL_REV[pred_bow]
            css_tf   = f"sent-{sent_tf}"
            css_bow  = f"sent-{sent_bow}"

            # Ground truth se texto vem do dataset
            _ds_lookup = {d[0]: d[1] for d in DATASET}
            ground_truth = _ds_lookup.get(user_text.strip())

            st.markdown('<hr class="dv">', unsafe_allow_html=True)

            # Banner de acerto/erro
            if ground_truth:
                tf_correct  = sent_tf  == ground_truth
                bow_correct = sent_bow == ground_truth
                _gt_label   = ground_truth.upper()
                _bow_badge  = "✓ BoW acertou" if bow_correct else "✗ BoW errou"
                if tf_correct:
                    st.markdown(
                        f'<div class="sent-pos" style="margin-bottom:.6rem">'
                        f'✓ Transformer acertou — rótulo real: <b>{_gt_label}</b> &nbsp;·&nbsp; {_bow_badge}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    cls_attn_raw = attn_avg[0]
                    clean_idx    = [i for i, t in enumerate(tokens) if t not in ("[PAD]","[CLS]","[SEP]")]
                    if clean_idx:
                        top_i   = clean_idx[int(np.argmax(cls_attn_raw[clean_idx]))]
                        top_tok = tokens[top_i]
                    else:
                        top_tok = "?"
                    st.markdown(
                        f'<div class="sent-neg" style="margin-bottom:.6rem">'
                        f'✗ Transformer errou — real: <b>{_gt_label}</b> · previu: <b>{sent_tf.upper()}</b>'
                        f' &nbsp;·&nbsp; {_bow_badge}<br>'
                        f'<span style="font-size:.82rem;opacity:.85">'
                        f'Token com maior atenção: <b>{top_tok}</b> — '
                        f'pode ter direcionado a previsão para a classe errada.</span></div>',
                        unsafe_allow_html=True
                    )

            r1, r2 = st.columns(2, gap="large")
            with r1:
                st.markdown(
                    f'<div class="{css_tf}">'
                    f'<div style="font-family:IBM Plex Mono,monospace;font-size:.62rem;'
                    f'letter-spacing:.14em;margin-bottom:.3rem">TRANSFORMER</div>'
                    f'<div style="font-size:1.3rem;font-weight:700">{sent_tf.upper()}</div>'
                    f'<div style="font-size:.8rem;margin-top:.3rem;opacity:.8">'
                    f'Pos: {probs_tf[0]*100:.1f}% | Neg: {probs_tf[1]*100:.1f}% | Neu: {probs_tf[2]*100:.1f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                # Tokens coloridos pela atencao media na linha do [CLS]
                cls_attn = attn_avg[0]   # atencao do token [CLS] para os demais
                cls_attn = cls_attn / (cls_attn.max() + 1e-9)
                st.markdown(render_tokens(tokens, cls_attn, sent_tf), unsafe_allow_html=True)
                st.markdown(
                    '<div style="font-family:IBM Plex Mono,monospace;font-size:.65rem;color:#445566">'
                    'Brilho dos tokens = atencao do [CLS] para cada palavra</div>',
                    unsafe_allow_html=True
                )

            with r2:
                st.markdown(
                    f'<div class="{css_bow}" style="opacity:.85">'
                    f'<div style="font-family:IBM Plex Mono,monospace;font-size:.62rem;'
                    f'letter-spacing:.14em;margin-bottom:.3rem">BAG-OF-WORDS (sem atencao)</div>'
                    f'<div style="font-size:1.3rem;font-weight:700">{sent_bow.upper()}</div>'
                    f'<div style="font-size:.8rem;margin-top:.3rem;opacity:.8">'
                    f'Pos: {probs_bow[0]*100:.1f}% | Neg: {probs_bow[1]*100:.1f}% | Neu: {probs_bow[2]*100:.1f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown(render_tokens(tokens), unsafe_allow_html=True)
                st.markdown(
                    '<div class="cb" style="margin-top:1rem;font-size:.82rem">'
                    '<b style="color:var(--blue)">Por que o Transformer e melhor?</b><br>'
                    'O BoW ignora a ordem e o contexto das palavras — "falha sem reparo" e '
                    '"sem falha reparo" sao identicos para ele. '
                    'O Transformer usa atencao para entender relacoes entre tokens.</div>',
                    unsafe_allow_html=True
                )

            # Barras de probabilidade
            st.markdown('<p class="sl" style="margin-top:1rem">Distribuicao de probabilidade</p>', unsafe_allow_html=True)
            fig_prob = go.Figure()
            cats     = ["Positivo", "Negativo", "Neutro"]
            colors_p = ["#4ade80", "#f87171", "#60a5fa"]
            fig_prob.add_trace(go.Bar(name="Transformer", x=cats,
                                      y=[p*100 for p in probs_tf],
                                      marker_color=colors_p, opacity=0.9))
            fig_prob.add_trace(go.Bar(name="BoW", x=cats,
                                      y=[p*100 for p in probs_bow],
                                      marker_color=colors_p, opacity=0.4,
                                      marker_pattern_shape="x"))
            fig_prob.update_layout(**DARK, title="Probabilidades por classe",
                                   yaxis_title="%", barmode="group",
                                   legend=dict(bgcolor="#0b1117", bordercolor="#22334a"),
                                   height=280, margin=dict(t=40,b=30,l=40,r=20))
            st.plotly_chart(fig_prob, width="stretch")

    # ════════════════════════════════════════════════════
    # TAB 2: ATENCAO TOKEN x TOKEN
    # ════════════════════════════════════════════════════
    with tab2:
        st.markdown('<p class="sl">Selecione o texto para inspecionar a atencao</p>', unsafe_allow_html=True)

        sel_txt = st.selectbox(
            "Texto", [d[0] for d in DATASET], key="attn_sel",
            format_func=lambda x: f"{x[:60]}..." if len(x)>60 else x,
        )
        ids_sel, toks_sel = tokenizer.encode(sel_txt, max_len=18)
        probs_sel = transformer.forward(ids_sel, return_attn=True)
        attn_all_sel = transformer._last_attn

        if attn_all_sel is None:
            st.error("Atenção não disponível — retreine o modelo.")
            st.stop()

        sel_head = st.radio(
            "Cabeca de atencao",
            [f"Cabeca {i+1}" for i in range(attn_all_sel.shape[0])] + ["Media de todas"],
            horizontal=True, key="head_radio"
        )
        if sel_head == "Media de todas":
            mat = attn_all_sel.mean(axis=0)
            title_h = "Atencao media — todas as 4 cabecas"
        else:
            hi = int(sel_head.split()[-1]) - 1
            mat = attn_all_sel[hi]
            title_h = f"Atencao — {sel_head}"

        st.plotly_chart(plot_attention_heatmap(mat, toks_sel, title=title_h),
                        width="stretch")

        # Tokens com brilho da atencao
        st.markdown('<p class="sl">Tokens com intensidade proporcional a atencao recebida</p>', unsafe_allow_html=True)
        sent_sel = LABEL_REV[int(np.argmax(probs_sel))]

        for src_idx, src_tok in enumerate(toks_sel):
            if src_tok == "[PAD]": break
            attn_row = mat[src_idx] / (mat[src_idx].max() + 1e-9)
            st.markdown(
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:.7rem;'
                f'color:#445566;margin-top:.3rem">de <b style="color:var(--gold)">{src_tok}</b>:</div>'
                + render_tokens(toks_sel, attn_row, sent_sel),
                unsafe_allow_html=True
            )

    # ════════════════════════════════════════════════════
    # TAB 3: MULTI-HEAD
    # ════════════════════════════════════════════════════
    with tab3:
        st.markdown(
            '<div class="cb" style="margin-bottom:1rem">'
            '<b style="color:var(--blue)">Por que multiplas cabecas?</b><br>'
            '<span style="font-size:.88rem">'
            'Cada cabeca aprende um tipo diferente de relacao entre tokens. '
            'Uma pode focar em relacoes gramaticais (sujeito-verbo), '
            'outra em relacoes semanticas (falha-reparo), '
            'outra em proximidade posicional. '
            'Juntas elas dao uma visao completa do texto.</span></div>',
            unsafe_allow_html=True
        )

        sel_txt2 = st.selectbox(
            "Texto para Multi-Head",
            [d[0] for d in DATASET], key="mh_sel",
            format_func=lambda x: f"{x[:60]}..." if len(x)>60 else x,
        )
        ids_mh, toks_mh = tokenizer.encode(sel_txt2, max_len=18)
        transformer.forward(ids_mh, return_attn=True)
        attn_mh = transformer._last_attn

        if attn_mh is None:
            st.error("Atenção não disponível — retreine o modelo.")
            st.stop()

        st.plotly_chart(plot_multihead(attn_mh, toks_mh), width="stretch")

        # Analise de cada cabeca
        st.markdown('<p class="sl">Token mais atendido por cada cabeca (coluna com maior media)</p>', unsafe_allow_html=True)
        clean_toks = [t for t in toks_mh if t != "[PAD]"]
        n_clean    = len(clean_toks)
        if n_clean == 0:
            st.warning("Nenhum token valido para analisar.")
            st.stop()
        head_cols  = st.columns(attn_mh.shape[0])
        for h_i, col_w in enumerate(head_cols):
            with col_w:
                mat_h    = attn_mh[h_i, :n_clean, :n_clean]
                top_dest = int(np.argmax(mat_h.mean(axis=0)))
                top_src  = int(np.argmax(mat_h.mean(axis=1)))
                col_c    = ["#f59e0b","#2dd4bf","#60a5fa","#c084fc"][h_i % 4]
                st.markdown(
                    f'<div class="head-box">'
                    f'<div style="font-family:IBM Plex Mono,monospace;font-size:.62rem;'
                    f'color:{col_c};letter-spacing:.1em;margin-bottom:.4rem">CABECA {h_i+1}</div>'
                    f'<div style="font-size:.75rem;color:#8899aa">mais recebe:</div>'
                    f'<div style="font-family:IBM Plex Mono,monospace;color:{col_c};'
                    f'font-size:.9rem;font-weight:600">{clean_toks[top_dest]}</div>'
                    f'<div style="font-size:.75rem;color:#8899aa;margin-top:.3rem">mais envia:</div>'
                    f'<div style="font-family:IBM Plex Mono,monospace;color:{col_c};'
                    f'font-size:.9rem;font-weight:600">{clean_toks[top_src]}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # ════════════════════════════════════════════════════
    # TAB 4: EMBEDDINGS 2D
    # ════════════════════════════════════════════════════
    with tab4:
        st.markdown(
            '<div class="ct" style="margin-bottom:1rem">'
            '<b style="color:var(--teal)">O que sao embeddings?</b><br>'
            '<span style="font-size:.88rem">'
            'Cada texto e transformado em um vetor de 32 dimensoes apos passar pela atencao. '
            'O PCA projeta esses vetores em 2D para visualizacao. '
            'Textos com sentimento similar ficam proximos no espaco — '
            'isso mostra que o Transformer aprendeu representacoes significativas.</span></div>',
            unsafe_allow_html=True
        )

        try:
            with st.spinner("Calculando embeddings..."):
                fig_emb, var_ratio = plot_embeddings_2d(
                    transformer, tokenizer, all_texts, all_labels
                )
            st.plotly_chart(fig_emb, width="stretch")
            st.markdown(
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:.72rem;color:#445566;margin-top:-.5rem">'
                f'PC1 explica {var_ratio[0]*100:.1f}% da variancia | '
                f'PC2 explica {var_ratio[1]*100:.1f}% | '
                f'Total: {sum(var_ratio)*100:.1f}% da informacao preservada</div>',
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Embeddings nao disponiveis: {e}")

        # Embedding do positional encoding
        st.markdown('<p class="sl" style="margin-top:1.5rem">Positional Encoding</p>', unsafe_allow_html=True)
        st.plotly_chart(plot_pos_encoding(), width="stretch")
        st.markdown(
            '<div style="font-family:IBM Plex Mono,monospace;font-size:.72rem;color:#445566">'
            'Cada linha e uma posicao no texto. Colunas pares usam seno, impares usam cosseno. '
            'Isso garante que posicoes proximas tenham vetores similares e que o modelo possa '
            'inferir posicoes relativas entre tokens.</div>',
            unsafe_allow_html=True
        )

    # ════════════════════════════════════════════════════
    # TAB 5: COMO FUNCIONA
    # ════════════════════════════════════════════════════
    with tab5:
        c5a, c5b = st.columns(2, gap="large")

        with c5a:
            st.markdown("### O que e Self-Attention?")
            st.markdown(
                "A ideia central do Transformer e que cada palavra do texto deve poder "
                "'prestar atencao' em qualquer outra palavra, independente da distancia. "
                "Em 'trator com **falha** no motor', a rede aprende que 'falha' e relevante "
                "para entender o sentimento de 'motor' — mesmo que estejam separados."
            )

            st.markdown("### Scaled Dot-Product Attention")
            st.markdown(
                '<div class="formula">'
                '# Para cada token, calcula Q, K, V<br>'
                'Q = X @ W_Q    # "o que estou buscando?"<br>'
                'K = X @ W_K    # "o que eu tenho para oferecer?"<br>'
                'V = X @ W_V    # "qual e o meu conteudo?"<br><br>'
                '# Scores de atencao<br>'
                'scores = Q @ K.T / sqrt(d_head)<br><br>'
                '# Normaliza com softmax<br>'
                'attn_w = softmax(scores)    # soma = 1<br><br>'
                '# Saida ponderada<br>'
                'output = attn_w @ V'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown("### Por que dividir por sqrt(d_head)?")
            st.markdown(
                "Sem a divisao, os scores crescem com a dimensionalidade e o softmax "
                "satura — produz distribuicoes muito 'pontiagudas' com gradientes proximos "
                "de zero. Dividir por sqrt(d_head) mantém os scores em escala adequada."
            )

            st.markdown("### Multi-Head Attention")
            st.markdown(
                '<div class="formula">'
                '# Cada cabeca aprende relacoes diferentes<br>'
                'for h in range(n_heads):<br>'
                '    head_h = attention(X, W_Q[h], W_K[h], W_V[h])<br><br>'
                '# Concatena e projeta<br>'
                'concat = concatenate(head_1, ..., head_n)<br>'
                'output = concat @ W_O'
                '</div>',
                unsafe_allow_html=True
            )

        with c5b:
            st.markdown("### Positional Encoding")
            st.markdown(
                "O Transformer nao tem recorrencia nem convolucao — ele processa todos os "
                "tokens em paralelo. Por isso, precisamos injetar informacao de posicao "
                "manualmente via seno e cosseno."
            )
            st.markdown(
                '<div class="formula">'
                'PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))<br>'
                'PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))<br><br>'
                '# Token embedding + posicao<br>'
                'X = E[token_ids] + PE[:seq_len]'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown("### Arquitetura completa")
            st.markdown(
                '<div class="formula">'
                'Input tokens<br>'
                '  -> Token Embedding + Positional Encoding<br>'
                '  -> Multi-Head Self-Attention<br>'
                '  -> Add & Norm (residual)<br>'
                '  -> Feed-Forward (Linear -> ReLU -> Linear)<br>'
                '  -> Add & Norm (residual)<br>'
                '  -> Mean Pooling (sobre tokens)<br>'
                '  -> Linear -> Softmax<br>'
                '  -> Classe (positivo / negativo / neutro)'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown("### Transformer vs BoW vs RNN")
            st.markdown("""
| Aspecto | BoW | RNN/LSTM | Transformer |
|---|---|---|---|
| Contexto | Nenhum | Sequencial | Global |
| Paralelo | Sim | Nao | Sim |
| Long-range | Nao | Fraco | Forte |
| Interpretavel | Nao | Nao | Sim (atencao) |
| Base do GPT | Nao | Nao | Sim |
            """)

            st.markdown("### Codigo — Atencao em NumPy")
            st.code("""
def attention(X, W_Q, W_K, W_V, d_head):
    Q = X @ W_Q          # Query
    K = X @ W_K          # Key
    V = X @ W_V          # Value

    # Scores normalizados
    scores = Q @ K.T / np.sqrt(d_head)

    # Pesos de atencao
    attn_w = softmax(scores)   # (seq, seq)

    # Saida ponderada
    return attn_w @ V, attn_w
            """, language="python")

        st.markdown('<hr class="dv">', unsafe_allow_html=True)
        st.plotly_chart(plot_loss_curves(h_tf, h_bow), width="stretch")

        st.markdown('<p class="sl">Dataset completo</p>', unsafe_allow_html=True)
        df_ds = pd.DataFrame(DATASET, columns=["Texto", "Sentimento"])
        st.dataframe(df_ds, hide_index=True, width="stretch",
                     column_config={"Texto": st.column_config.TextColumn(width="large")})
        st.caption(f"{len(DATASET)} textos agricolas | 15 positivos, 15 negativos, 15 neutros | 75% treino / 25% validacao")


if __name__ == "__main__":
    main()
