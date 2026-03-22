# AttentionMind — Transformers do Zero
https://attentionmind-emhmu7x2xkrrnplfixapx6.streamlit.app/

<img width="1874" height="824" alt="image" src="https://github.com/user-attachments/assets/bbcd0ef8-7935-4553-90eb-da4e4e73741f" />

Demonstração educativa do **mecanismo de Atenção e Transformers** implementados em **NumPy puro**. Classifica sentimento de textos agrícolas (ordens de serviço, relatórios de campo) e visualiza o que a rede "está olhando" em cada palavra.

> Quinto projeto da série **ML Educativo**.

---

## Conceitos demonstrados

| Conceito | Descrição |
|---|---|
| **Self-Attention** | Cada token "olha" para todos os outros e decide quanto atentar em cada um |
| **Q, K, V** | Query, Key, Value — mecanismo de busca aprendível |
| **Scaled Dot-Product** | `softmax(QK^T / sqrt(d)) * V` — fórmula central do Transformer |
| **Multi-Head Attention** | Múltiplas cabeças aprendem relações diferentes em paralelo |
| **Positional Encoding** | Seno/cosseno para injetar informação de posição |
| **Residual + LayerNorm** | Estabilização do treinamento em redes profundas |
| **Embeddings** | Representações vetoriais de tokens no espaço contínuo |
| **PCA de embeddings** | Visualização 2D do espaço aprendido pela rede |

---

## Funcionalidades

- **Classificação interativa** — classifica qualquer texto agrícola com o Transformer e com BoW (comparação)
- **Tokens coloridos por atenção** — brilho de cada palavra = quanto o [CLS] atendeu a ela
- **Heatmap token × token** — matriz de atenção completa, cabeça selecionável
- **Multi-head grid** — todas as 4 cabeças lado a lado com análise de foco
- **Embeddings 2D** — PCA dos vetores pós-atenção, agrupados por sentimento
- **Positional encoding** — heatmap visualizando seno/cosseno por posição e dimensão
- **Comparação Transformer vs BoW** — curvas de acurácia e loss em tempo real
- **Código exposto** — implementação NumPy comentada na aba "Como Funciona"

---

## Sem dependências pesadas

O Transformer foi implementado **do zero em NumPy** — sem PyTorch, TensorFlow ou HuggingFace para o modelo. Apenas:

```
streamlit, numpy, pandas, plotly, scikit-learn
```

---

## Como executar

```bash
git clone https://github.com/Victormartinsilva/AttentionMind.git
cd AttentionMind
pip install -r requirements.txt
streamlit run app.py
```

---

## Arquitetura do Transformer

```
Input tokens
  -> Token Embedding [vocab, 32] + Positional Encoding
  -> Multi-Head Self-Attention (4 cabeças × d_head=8)
  -> Add & Norm (residual connection)
  -> Feed-Forward (32 -> 64 -> 32, ReLU)
  -> Add & Norm
  -> Mean Pooling sobre tokens
  -> Linear [32, 3] -> Softmax
  -> Classe: positivo / negativo / neutro
```

---

## Série ML Educativo

| # | Projeto | Conceito |
|---|---|---|
| 1 | Jogo da Velha | Q-Learning |
| 2 | Diagnóstico de Plantas | Árvore de Decisão |
| 3 | TractorMind | Regras de Associação (Apriori) |
| 4 | [NeuralMind](https://github.com/Victormartinsilva/REDE_NEURAL) | Rede Neural MLP do Zero |
| 5 | **AttentionMind** | **Transformers e Self-Attention** |

**← Projeto anterior:** [NeuralMind](https://github.com/Victormartinsilva/REDE_NEURAL)

---

**Autor:** Victor Martin Silva
**Licença:** MIT
