# FarmTech Solutions - Fase 6

Projeto de visão computacional para comparar YOLOv5n ajustado (30 e 60 épocas), YOLOv5n COCO sem ajuste e CNN treinada do zero. Classes: garrafa e xícara de café. O dataset contém 80 imagens públicas, separadas em 64/8/8.

## Integrantes e acesso

- Grupo: PREENCHER
- Nome completo e RM de cada integrante: PREENCHER
- Repositório público: PREENCHER URL REAL
- Colab compartilhado com acesso ao dataset: PREENCHER URL REAL
- Vídeo YouTube não listado (até 5 minutos): PREENCHER URL REAL

## Documentação principal

- [Notebook completo para Colab](NomeCompleto_rmXXXXX_pbl_fase6.ipynb): passo a passo de treinamento, validação e teste. Renomear com nome completo/RM, executar e salvar saídas antes da entrega.
- [Resultados originais verificados](Fase6_FarmTech_Resultados_Verificados.ipynb): análise executada dos resultados fornecidos e métricas corrigidas.
- [Relatório](Fase6_FarmTech_Trabalho_Final.pdf): resultados, discussão e limitações.
- [Pendências de entrega](ANTES_DE_ENTREGAR.md).
- [Guia Make Sense AI](GUIA_MAKE_SENSE.md).
- [Roteiro do vídeo](ROTEIRO_VIDEO.md).

## Como executar

1. Enviar `Fase6_FarmTech_Corrigido.zip` para `Meu Drive/FarmTech_Fase6/`.
2. Abrir o notebook completo no Colab; conferir o caminho `ZIP_DRIVE`.
3. Executar as células em ordem. A instalação deve preceder imports de NumPy/Pandas/Pillow; reiniciar a sessão se houver imports anteriores incompatíveis.
4. Os resultados novos ficam em `execucoes_novas/` e são copiados para o Drive. Salvar também o notebook com suas saídas.
5. Atualizar a discussão usando os resultados da execução que será entregue.

## Organização

`openimages/dataset`: imagens e rótulos; `openimages/runs`: pesos e curvas originais; `openimages/logs`: logs originais; `openimages/cnn`: modelo/histórico CNN; `openimages/figures`: gráficos e testes; `evidencias_originais`: arquivos anteriores às correções; scripts Python: implementação; `project_config.py`: caminhos comuns.

## Estado e procedência

Dados e caixas: Open Images; atribuição individual em `openimages/attribution.csv`. As anotações atuais não comprovam a etapa Make Sense AI. Fotos próprias não são obrigatórias no texto fornecido. O pacote não contém vídeo publicado nem repositório criado. As Entregas 1 e 2 são o escopo; Ir Além é opcional e não foi implementado.

O F1 macro foi corrigido; E60 melhorou E30, mas não superou COCO em mAP50/F1 macro no teste original. Não se deve comparar acurácia da CNN diretamente com mAP dos detectores.

Código YOLO externo: [Ultralytics YOLOv5 v7.0](https://github.com/ultralytics/yolov5/tree/v7.0). Compatibilidade de checkpoints: [documentação PyTorch](https://docs.pytorch.org/docs/2.6/notes/serialization.html).
