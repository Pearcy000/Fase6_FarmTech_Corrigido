# Rotulação no Make Sense AI

Esta etapa deve ser realizada de verdade; não foi substituída por uma declaração de uso.

1. Abrir https://www.makesense.ai/ e carregar as 64 imagens de `openimages/dataset/images/train` para detecção.
2. Criar as classes nesta ordem: `bottle` (0), `coffee_cup` (1).
3. Desenhar/revisar caixas para todas as instâncias visíveis das duas classes. Os rótulos Open Images podem servir de referência. Se houver opção de importar os rótulos existentes, revisá-los e documentar que houve importação/revisão, sem afirmar anotação original do zero.
4. Exportar no formato YOLO; conferir associação nome de imagem/nome de TXT e IDs das classes. Guardar export, prints do editor e data no Drive.
5. Substituir os rótulos de treino revisados e executar `validate_dataset.py`. Conferir visualmente caixas e limites; o validador não detecta toda omissão de objeto.
6. Preservar validação/teste; revisar suas referências se necessário e documentar as alterações antes de avaliar.
7. Se qualquer caixa mudar, treinar e avaliar novamente. As métricas históricas do pacote descrevem os rótulos originais e não podem ser atribuídas a um dataset modificado.

Não inventar prints nem atribuir a rotulação a uma ferramenta que não foi usada. Se a instituição autorizar outra ferramenta, guardar a autorização específica.
