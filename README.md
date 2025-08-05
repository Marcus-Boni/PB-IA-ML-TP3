# Projeto de Classificação Binária: Detecção de Minas com PCA e Árvores de Decisão

## 📝 Visão Geral

Este projeto demonstra a construção de um pipeline de Machine Learning para uma tarefa de classificação binária. O objetivo é criar um modelo capaz de distinguir entre minas e rochas com base em dados de sinais de sonar. Para isso, são aplicadas técnicas avançadas como Análise de Componentes Principais (PCA) para redução de dimensionalidade e `GridSearchCV` para otimização de hiperparâmetros de um classificador de Árvore de Decisão.

## 🎯 Objetivo

Proporcionar uma experiência prática completa em técnicas de Machine Learning, cobrindo as seguintes etapas:

  * **Redução de Dimensionalidade:** Aplicar PCA para otimizar o conjunto de features.
  * **Modelagem e Treinamento:** Desenvolver e treinar um modelo de Árvore de Decisão.
  * **Validação e Otimização:** Utilizar Validação Cruzada e `GridSearch` para encontrar o modelo mais robusto e com melhor performance.
  * **Prevenção de Overfitting:** Aplicar a técnica de poda (pruning) na árvore de decisão.
  * **Avaliação de Performance:** Analisar a eficiência do classificador final utilizando um conjunto completo de métricas, como Acurácia, Precisão, Recall, F1-Score, Especificidade e a Curva ROC.

## 💾 Base de Dados

O conjunto de dados utilizado é o **"Sonar, Mines vs. Rocks"**, que contém 208 amostras. Cada amostra consiste em 60 features que representam a energia de um sinal de sonar em diferentes janelas de frequência. A variável alvo indica se o objeto detectado é uma Rocha (`R`) ou uma Mina (`M`).

  * **Fonte:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+\(Sonar,+Mines+vs.+Rocks\))
  * **Arquivo:** `sonar_dataset.csv`

## 🛠️ Tecnologias e Bibliotecas

O projeto foi desenvolvido inteiramente em Python (\>3.9) e utiliza as seguintes bibliotecas:

  * **Pandas:** Para manipulação e análise dos dados.
  * **NumPy:** Para operações numéricas eficientes.
  * **Scikit-learn:** Para a implementação do pipeline de Machine Learning (PCA, Árvore de Decisão, `GridSearchCV`, métricas de avaliação).
  * **Matplotlib** e **Seaborn:** Para a visualização de dados e resultados, como a matriz de confusão e a curva ROC.

## methodological Abordagem Metodológica

O fluxo de trabalho do projeto foi estruturado da seguinte forma:

1.  **Preparação dos Dados:**

      * Carregamento do dataset `sonar_dataset.csv`.
      * Separação das features (X) e da variável alvo (y).
      * Codificação da variável alvo de categórica ('R', 'M') para numérica (0, 1).
      * Divisão dos dados em conjuntos de treino (80%) e teste (20%) de forma estratificada para manter a proporção das classes.

2.  **Redução de Dimensionalidade com PCA:**

      * Padronização (scaling) dos dados de treino para que todas as features tenham média 0 e desvio padrão 1, um pré-requisito para a eficácia do PCA.
      * Aplicação do PCA para identificar o número de componentes principais que retêm **95% da variância** dos dados originais, reduzindo o número de features de 60 para 28.

3.  **Modelagem e Otimização:**

      * **Baseline:** Treinamento de uma Árvore de Decisão com hiperparâmetros padrão para estabelecer uma performance inicial.
      * **Busca Hiperparamétrica:** Utilização do `GridSearchCV` com validação cruzada (10-fold) para testar exaustivamente uma combinação de hiperparâmetros, incluindo `criterion`, `max_depth` e `ccp_alpha` (para poda).
      * **Seleção do Melhor Modelo:** O modelo com a melhor combinação de hiperparâmetros encontrada pelo `GridSearch` foi selecionado para a avaliação final.

4.  **Avaliação Final:**

      * O modelo otimizado foi avaliado no conjunto de teste (dados nunca vistos).
      * Foram calculadas diversas métricas de classificação para obter uma visão completa da performance do modelo.

## 📊 Resultados e Análise

O modelo final demonstrou uma performance robusta e eficiente, superando significativamente a linha de base.

| Métrica                 | Resultado no Conjunto de Teste | Interpretação                                                                                             |
| ----------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------- |
| **Acurácia** | `85.71%`                       | O modelo classifica corretamente 85.71% dos objetos.                                                      |
| **Precisão (para Mina)** | `85.0%`                        | Quando o modelo prevê "Mina", ele está correto em 85% dos casos.                                          |
| **Recall (Sensibilidade)** | `89.5%`                        | O modelo identifica corretamente 89.5% de todas as minas reais, minimizando o risco de falsos negativos. |
| **F1-Score** | `87.2%`                        | Excelente equilíbrio entre precisão e recall.                                                             |
| **Especificidade** | `82.6%`                        | O modelo identifica corretamente 82.6% das rochas, mostrando uma baixa taxa de alarmes falsos.            |
| **AUC (Curva ROC)** | `0.86`                         | Excelente capacidade do modelo em distinguir entre as classes "Mina" e "Rocha".                          |

### Matriz de Confusão

*(Sugestão: Faça upload das imagens geradas, como a matriz de confusão e a curva ROC, para um serviço como o Imgur e insira os links aqui)*

### Curva ROC

### Conclusão sobre a Eficiência

O classificador desenvolvido é **altamente eficiente**. A combinação de PCA e a otimização com `GridSearchCV` transformaram um modelo mediano em uma solução robusta e precisa. O **alto recall (sensibilidade)** é particularmente importante para este problema, pois a falha em detectar uma mina (falso negativo) é muito mais crítica do que um alarme falso (falso positivo). O modelo final equilibra bem essa necessidade com uma boa taxa de especificidade, tornando-o prático e confiável.

## 🚀 Como Executar o Projeto

Para executar o notebook e replicar os resultados, siga os passos abaixo:

1.  **Clone o repositório:**

    ```bash
    git clone https://github.com/Marcus-Boni/PB-IA-ML-TP3.git
    cd PB-IA-ML-TP3
    ```

2.  **Crie um ambiente virtual (recomendado):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3.  **Instale as dependências:**

    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn jupyter
    ```

4.  **Execute o Jupyter Notebook:**

      * Certifique-se de que o arquivo `sonar_dataset.csv` está no mesmo diretório.
      * Inicie o Jupyter:
        ```bash
        jupyter notebook
        ```
      * Abra o arquivo `sonar_classification_project.ipynb` e execute as células.

### 👥 Contribuições

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request


---

⭐ **Se este projeto foi útil para você, considere dar uma estrela!** ⭐
