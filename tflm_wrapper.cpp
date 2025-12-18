#include <cstdio>
#include "pico/stdlib.h"

// -------------------------------------------------------------------
// TensorFlow Lite Micro (via pico-tflmicro)
// -------------------------------------------------------------------
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Modelo convertido em array C (float32 - gerado no Colab)
#include "wine_mlp_float.h"   // <--- Mude aqui se o nome do seu header for diferente

// API em C que será chamada pelo main.c
#include "tflm_wrapper.h"

// -------------------------------------------------------------------
// Objetos estáticos do TFLM
// -------------------------------------------------------------------
namespace {

constexpr int kTensorArenaSize = 16 * 1024;  // 16 KB (suficiente para o modelo Wine float32)
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

static tflite::MicroErrorReporter micro_error_reporter;
static tflite::ErrorReporter* error_reporter = &micro_error_reporter;

static const tflite::Model* model = nullptr;

static tflite::MicroMutableOpResolver<4> resolver;

static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

}  // namespace

// -------------------------------------------------------------------
// Inicializa o modelo TFLM
// -------------------------------------------------------------------
int tflm_init_model(void) {
    // Carrega o modelo float32
    model = tflite::GetModel(wine_mlp_float_tflite);  // <--- Nome exato do array gerado pelo xxd
    if (model == nullptr) {
        printf("Erro: modelo nulo.\n");
        return -1;
    }

    // Registra as operações usadas pela MLP (Dense, ReLU, Softmax, Reshape)
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();
    resolver.AddReshape();

    // Cria o intérprete
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);

    interpreter = &static_interpreter;

    // Aloca tensores
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors falhou.\n");
        return -2;
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    if (!input_tensor || !output_tensor) {
        printf("Erro ao obter tensores de entrada/saída.\n");
        return -3;
    }

    printf("TFLM inicializado com sucesso.\n");
    printf("Dimensoes input: ");
    for (int i = 0; i < input_tensor->dims->size; i++) {
        printf("%d ", input_tensor->dims->data[i]);
    }
    printf("\nDimensoes output: ");
    for (int i = 0; i < output_tensor->dims->size; i++) {
        printf("%d ", output_tensor->dims->data[i]);
    }
    printf("\n");

    return 0;
}

// -------------------------------------------------------------------
// Executa uma inferência (13 features de entrada → 3 scores de saída)
// -------------------------------------------------------------------
int tflm_infer(const float in_features[13], float out_scores[3]) {
    if (!interpreter || !input_tensor || !output_tensor) {
        return -1;
    }

    // Copia as 13 features normalizadas para o tensor de entrada
    for (int i = 0; i < 13; i++) {
        input_tensor->data.f[i] = in_features[i];
    }

    // Executa a inferência
    if (interpreter->Invoke() != kTfLiteOk) {
        printf("Invoke falhou.\n");
        return -2;
    }

    // Lê as 3 probabilidades de saída
    for (int i = 0; i < 3; i++) {
        out_scores[i] = output_tensor->data.f[i];
    }

    return 0;
}