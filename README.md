# FarmTech Solutions — Fase 6: O começo da rede neural

Projeto acadêmico de visão computacional da FIAP. A solução demonstra a identificação de **garrafas** e **xícaras de café** e compara YOLOv5 personalizado, YOLOv5 pré-treinado sem personalização e uma CNN treinada do zero.

Este README introduz o projeto e orienta o acesso à solução. O passo a passo, o código, as avaliações e as conclusões estão nos notebooks, sem repetir aqui o relatório completo.

## Identificação

- **Grupo:** nome ainda não informado.

| Integrante | RM |
|---|---|
| Camila Duarte Ferreira | 569629 |
| João Luiz Bonim | 571052 |
| Gabriel Fileto Di Capua da Silva | 569179 |
- **Repositório público:** [Fase6_FarmTech_Corrigido](https://github.com/Pearcy000/Fase6_FarmTech_Corrigido).

O notebook principal usa o nome completo e o RM de Gabriel no padrão exigido pelo enunciado. O nome do grupo ainda precisa ser informado para adequar o nome do repositório.

## Acesse a solução

| Documento | Finalidade |
|---|---|
| [Notebook principal no Colab](https://colab.research.google.com/github/Pearcy000/Fase6_FarmTech_Corrigido/blob/main/GabrielFiletoDiCapuaDaSilva_rm569179_pbl_fase6.ipynb) · [Ver no GitHub](GabrielFiletoDiCapuaDaSilva_rm569179_pbl_fase6.ipynb) | Preparação do ambiente, conexão com o Drive, treinamento, validação, teste e comparação das abordagens. **As células de código ainda precisam ser executadas e salvas.** |
| [Resultados verificados](Fase6_FarmTech_Resultados_Verificados.ipynb) · [Abrir no Colab](https://colab.research.google.com/github/Pearcy000/Fase6_FarmTech_Corrigido/blob/main/Fase6_FarmTech_Resultados_Verificados.ipynb) | Análise dos resultados fornecidos, com células executadas e saídas salvas. Não substitui a execução do notebook principal. |
| [Relatório em PDF](Fase6_FarmTech_Trabalho_Final.pdf) | Discussão dos resultados, conclusões e limitações. |

Para reproduzir a solução, abra o notebook principal e siga suas células Markdown. Ele orienta a disponibilização dos arquivos no Google Drive e a configuração dos caminhos do dataset. O acesso aos dados no Drive ainda precisa ser garantido para a correção.

## Entregas obrigatórias

**Entrega 1 — YOLO personalizado:** 80 imagens, com 40 por classe e divisão de 32/4/4 por classe para treino, validação e teste. O projeto inclui experimentos de 30 e 60 épocas, métricas e imagens de teste processadas.

**Entrega 2 — comparação:** YOLO personalizado, YOLO pré-treinado COCO e CNN treinada do zero, considerando facilidade de integração, qualidade das previsões, tempo de treinamento e tempo de inferência. A análise completa está nos notebooks e no relatório. Métricas de classificação da CNN e de detecção do YOLO exigem interpretação distinta.

As opções **Ir Além** são facultativas e não foram implementadas.

## Vídeo demonstrativo

**Link do YouTube: ainda não informado.**

É exigido um vídeo de **até 5 minutos**, publicado como **não listado**, demonstrando o funcionamento da solução. Insira o link nesta seção antes da entrega. Consulte o [roteiro de apoio](ROTEIRO_VIDEO.md).

## Organização

- `openimages/dataset/`: imagens e rótulos em treino, validação e teste.
- `openimages/runs/`, `openimages/test_runs/` e `openimages/cnn/`: modelos, métricas e evidências.
- `openimages/figures/`: gráficos e exemplos de detecção.
- Scripts Python e notebooks: implementação e reprodução.
- `evidencias_originais/`: resultados anteriores à revisão.

Imagens e anotações existentes vêm do Open Images; a atribuição está em [openimages/attribution.csv](openimages/attribution.csv). **O material atual não comprova a rotulação no Make Sense AI exigida pelo enunciado.** Essa etapa precisa ser realizada e documentada; consulte o [guia de rotulação](GUIA_MAKE_SENSE.md).

## Antes da entrega

- Informar o nome do grupo e adequar o nome do repositório.
- Executar o notebook principal, salvar código e saídas, avaliar os resultados e garantir acesso ao dataset e aos rótulos no Drive.
- Realizar e comprovar a rotulação no Make Sense AI.
- Inserir o link do vídeo não listado.

Veja a [lista completa de conferência](ANTES_DE_ENTREGAR.md). Após a submissão no portal da FIAP, não faça novos commits: o enunciado considera alterações posteriores à data de entrega como entrega fora do prazo.
