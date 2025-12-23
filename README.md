# SINALIZE

**SIN**alise - Modelo de previsão de falhas para transformadores do Sistema Interligado Nacional

## Resumo

Este trabalho apresenta a implementação de um modelo preditivo para antecipação de falhas em transformadores da rede básica do Sistema Interligado Nacional (SIN). Utilizando algoritmos de Random Forest e técnicas de balanceamento de dados, a ferramenta, denominada **SINALIZE**, foi capaz de prever falhas em diferentes horizontes temporais, com base em características históricas de manutenção, falhas e utilização dos equipamentos, demonstrando a viabilidade da aplicação de técnicas de aprendizado de máquina para manutenção preditiva em sistemas elétricos de potência, contribuindo para a melhoria da confiabilidade, segurança e redução de custos operacionais.

Link para o artigo:
[SINALIZE - Modelo de previsão de falhas em transformadores do SIN (PDF)](SINALIZE-modelo-de-previsao-de%20falhas-em-transformadores-do-SIN.pd)

## Objetivos

- Prever falhas em transformadores da rede básica do SIN
- Otimizar estratégias de manutenção preventiva
- Reduzir custos operacionais e melhorar confiabilidade do sistema 
- Fornecer rankings de risco para priorização de recursos de manutenção de equipamentos

## Tecnologias Utilizadas

- **Python 3.8+**
- **Random Forest** (scikit-learn)
- **SMOTE** para balanceamento de dados
- **Flask** para API REST
- **Pandas** para manipulação de dados
- **NumPy** para computação numérica
- **Pytest** para testes unitários


## Estrutura do Projeto

```
├── src/                       # Código fonte do modelo preditivo
│   ├── 01-preparar-dados-preditivo.py
│   ├── 02-treinar-modelo-preditivo.py
│   ├── 03-api-teste-modelo.py
│   └── provenance.ttl  # modelo de data provenance 
├── data/
│   ├── raw/                   # Dados brutos
│   └── gold/                  # Dados processados
├── models/
│   └── rf/                    # Modelos treinados
├── requirements.txt
├── README.md
└── SINALIZE-modelo-de-previsao-de\ falhas-em-transformadores-do-SIN.pd # PDF DO ARTIGO da disciplina 

```

## Como Usar

### 1. Instalação

```bash
git clone https://github.com/Rrg331/Sinalize.git
cd Sinalize
pip install -r requirements.txt
```

### 2. Preparar Dados

```bash
cd src
python 01-preparar-dados-preditivo.py
```

### 3. Treinar Modelo

```bash
python 02-treinar-modelo-preditivo.py
```

### 4. Executar API para realizar previsões

```bash
python 03-api-teste-modelo.py
```

A API estará disponível em `http://localhost:5001`


## Trabalhos futuros 

Integração com dados de meteorologia já disponíveis no repositório para criar feature de sazonalidades
Integração com dados de queimadas para criar features  de impactos de queimadas sazonais
Criação de  feature de manobras de equipamentos utilizando dados de supervisão atuais

