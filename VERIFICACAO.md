# Verificação da revisão

- Sintaxe de todos os scripts Python e células do notebook principal: aprovada.
- Dataset: validação executada com imagens, rótulos, limites, classes e hashes.
- Notebook de resultados: células de análise executadas e saídas atualizadas.
- F1 macro: recalculado dos dados originais por classe.
- Melhor época CNN: 37, conferida com acurácia e loss do histórico.
- Treinamento/inferência novos: não executados; PyTorch indisponível neste computador. Colab e acesso ao Drive ainda precisam de execução real.

```json
{
  "train": {
    "images": 64,
    "images_per_class": {
      "coffee_cup": 32,
      "bottle": 32
    },
    "boxes": {
      "1": 57,
      "0": 176
    }
  },
  "val": {
    "images": 8,
    "images_per_class": {
      "coffee_cup": 4,
      "bottle": 4
    },
    "boxes": {
      "1": 7,
      "0": 17
    }
  },
  "test": {
    "images": 8,
    "images_per_class": {
      "coffee_cup": 4,
      "bottle": 4
    },
    "boxes": {
      "1": 5,
      "0": 19
    }
  }
}
```
