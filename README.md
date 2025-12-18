# TinyML – Classificação do Dataset Wine no Raspberry Pi Pico W  
### Prática com Rede Neural Artificial (RNA) para Microcontroladores

![Banner da Aula](aula_sincrona.png)  
![Tarefa em Grupo](tarefa.png)

Este projeto implementa uma **Rede Neural Artificial (RNA)**, Perceptron Multicamadas (MLP), embarcada no **Raspberry Pi Pico W**, utilizando a biblioteca **TensorFlow Lite Micro (TFLM)** para executar inferência diretamente no microcontrolador — abordagem típica de **TinyML**.

Este código faz parte de um projeto que demonstra como treinar, converter e executar um modelo inteligente real em um dispositivo de recursos extremamente limitados. Como conteúdo complementar, o modelo foi treinado usando o Google Colab, o link do código está disponível em: https://colab.research.google.com/drive/1fPQJ3YzNQpezyzfeRzWv1-7KZR6QM3uM?usp=sharing

---

## 📌 Objetivos

- Demonstrar o fluxo completo de TinyML:  
  **Criação do modelo → Treinamento → Conversão → Deploy → Inferência embarcada**
- Normalizar dados embarcados de forma idêntica ao treinamento.
- Executar inferências usando TFLM. Biblioteca disponível em: https://github.com/raspberrypi/pico-tflmicro.git
- Construir e imprimir a **matriz de confusão** 3×3.
- Calcular a acurácia final diretamente no microcontrolador.
- Integrar código C/C++ ao TensorFlow Lite Micro via wrapper.
- Utilizar um dataset diferente do Iris (exemplo da aula), conforme requisito da tarefa.

---

## 🧠 Visão geral

A aplicação embarcada no Pico W:

1. Carrega um modelo **MLP (rede neural multicamadas)** treinado com o dataset Wine.
2. Aplica normalização padrão (média e desvio).
3. Executa inferência amostra por amostra (178).
4. Constrói a **matriz de confusão 3×3** (real × predito).
5. Calcula a acurácia final da rede.
6. Exibe tudo via USB/serial.

Essa prática permite que estudantes compreendam como modelos inteligentes podem ser executados em **microcontroladores**, base fundamental para aplicações TinyML e Edge AI.

---

## 🍷 Dataset Utilizado: Wine Recognition

O dataset escolhido foi o **Wine Recognition** (disponível no UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/109/wine).

- **Descrição**: Resultados de análise química de vinhos cultivados na mesma região da Itália, mas derivados de três cultivares diferentes.
- **Número de amostras**: 178
- **Número de features**: 13 (todas numéricas contínuas)
- **Classes**: 3 (class_0: 59 amostras, class_1: 71 amostras, class_2: 48 amostras)

**Features (13 atributos químicos)**:
1. Alcohol
2. Malic acid
3. Ash
4. Alcalinity of ash
5. Magnesium
6. Total phenols
7. Flavanoids
8. Nonflavanoid phenols
9. Proanthocyanins
10. Color intensity
11. Hue
12. OD280/OD315 of diluted wines
13. Proline

Esse dataset foi escolhido por ser tabular simples, diferente do Iris utilizado em aula, e por apresentar desafio moderado (classes 1 e 2 com sobreposição), permitindo demonstrar comportamentos reais de TinyML em hardware restrito.

---

## 🧠 Arquitetura da Rede Neural (MLP)

- Camada de entrada: 13 neurônios (13 features do Wine)
- Camada oculta 1: 16 neurônios (ativação ReLU)
- Camada oculta 2: 8 neurônios (ativação ReLU)
- Camada de saída: 3 neurônios (ativação Softmax)

![Diagrama da Rede Neural](diagrama_rede.png)

---

## 📊 Resultados

- **Treinamento (Google Colab)**: Acurácia ~98% no conjunto de teste.
- **Inferência embarcada (RP2040)**: Acurácia 87,08%.

A diferença é esperada devido à menor precisão de ponto flutuante no RP2040 (float32 vs float64 no PC), especialmente em datasets com sobreposição de classes.

![Matriz de Confusão - Colab](matriz_colab.png)  
![Curva de Loss](curvas_loss.png)  
![Curva de Acurácia](curvas_accuracy.png)  
![Saída Serial - RP2040](serial_rp2040.png)

---

## 📁 Organização dos arquivos

### `tiny_ml_02.c`
Aplicação principal em C.  
Responsável por:
- Inicializar o Pico W e o ambiente TFLM.  
- Normalizar cada amostra com `wine_means` e `wine_stds`.  
- Realizar inferências via `tflm_infer()`.  
- Construir a matriz de confusão.  
- Calcular a acurácia e imprimir os resultados.

### `tflm_wrapper.h` / `tflm_wrapper.cpp`
Wrapper em C/C++ para o TensorFlow Lite Micro. Forma uma camada de abstração que encapsula o TensorFlow Lite Micro, oferecendo funções simples para inicializar o modelo, passar entradas e pegar saídas, sem que você precise lidar diretamente com todos os detalhes internos da biblioteca.
- Configura a arena de tensores.  
- Carrega o modelo embarcado (`wine_mlp_float_tflite`).  
- Registra operações necessárias (Dense, ReLU, Softmax).  
- Expõe:
  - `tflm_init_model()`  
  - `tflm_infer(float input[13], float output[3])`

### `wine_mlp_float.h`
Modelo TFLite convertido para array C (`unsigned char[]`), contendo a rede neural MLP treinada previamente em Python.

### `wine_dataset.h`
Dataset Wine embarcado no firmware:
- `wine_features[178][13]`  
- `wine_labels[178]`

### `wine_normalization.h`
Estatísticas de normalização utilizadas:
- `wine_means[13]`  
- `wine_stds[13]`  
Esses valores replicam exatamente o StandardScaler do treinamento, garantindo consistência na inferência.

### `CMakeLists.txt`
Arquivo de build usando pico-sdk + TFLM:
- Configuração do projeto
- Inclusão do TensorFlow Lite Micro
- Compilação dos arquivos `.c` e `.cpp`
- Links com bibliotecas padrão do Pico

---

## 🔧 Como compilar o projeto

### 1. Instale o Pico SDK
Disponível em: https://github.com/raspberrypi/pico-sdk

### 2. Configure e compile
```bash
mkdir build
cd build
cmake ..
make -j4



## 🔗 Links Importantes

Google Colab (treinamento e geração do modelo):https://colab.research.google.com/drive/1fPQJ3YzNQpezyzfeRzWv1-7KZR6QM3uM?usp=sharing
Repositório base do professor (Iris):https://github.com/rmprates84/tiny_ml_iris
Dataset Wine (UCI):https://archive.ics.uci.edu/dataset/109/wine
Biblioteca pico-tflmicro:https://github.com/raspberrypi/pico-tflmicro