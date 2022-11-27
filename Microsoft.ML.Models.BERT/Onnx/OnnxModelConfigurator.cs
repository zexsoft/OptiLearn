using Microsoft.ML;
using Microsoft.ML.Models.BERT.Onnx;
using Microsoft.ML.Transforms.Onnx;
using System.Collections.Generic;

namespace Microsoft.ML.Models.BERT.Onnx
{
    public class OnnxModelConfigurator<TFeature> where TFeature : class
    {
        private readonly MLContext _mlContext;
        private readonly ITransformer _mlModel;

        public OnnxModelConfigurator(IOnnxModel onnxModel)
        {
            _mlContext = new MLContext();
            _mlModel = SetupMlNetModel(onnxModel);
        }

        public OnnxModelConfigurator(IOnnxModel onnxModel, Dictionary<string, int[]> shapeDict)
        {
            _mlContext = new MLContext();
            _mlModel = SetupMlNetModel(onnxModel, shapeDict);
        }

        private ITransformer SetupMlNetModel(IOnnxModel onnxModel, Dictionary<string, int[]> shapeDict = null)
        {
            bool hasGpu = false;

            var dataView = _mlContext.Data
                .LoadFromEnumerable(new List<TFeature>());

            OnnxScoringEstimator pipeline;

            if (shapeDict != null)
                pipeline = _mlContext.Transforms
                            .ApplyOnnxModel(modelFile: onnxModel.ModelPath, outputColumnNames: onnxModel.ModelOutput, inputColumnNames: onnxModel.ModelInput, shapeDictionary: shapeDict, gpuDeviceId: hasGpu ? 0 : (int?)null, fallbackToCpu: true);
            else
                pipeline = _mlContext.Transforms
                            .ApplyOnnxModel(modelFile: onnxModel.ModelPath, outputColumnNames: onnxModel.ModelOutput, inputColumnNames: onnxModel.ModelInput, gpuDeviceId: hasGpu ? 0 : (int?)null, fallbackToCpu: true);

            var mlNetModel = pipeline.Fit(dataView);

            return mlNetModel;
        }

        public PredictionEngine<TFeature, T> GetMlNetPredictionEngine<T>() where T : class, new()
        {
            return _mlContext.Model.CreatePredictionEngine<TFeature, T>(_mlModel);
        }

        public void SaveMLNetModel(string mlnetModelFilePath)
        {
            _mlContext.Model.Save(_mlModel, null, mlnetModelFilePath);
        }
    }
}
