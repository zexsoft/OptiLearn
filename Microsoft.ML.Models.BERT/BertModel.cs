using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.Models.BERT.Extensions;
using Microsoft.ML.Models.BERT.Input;
using Microsoft.ML.Models.BERT.Onnx;
using Microsoft.ML.Models.BERT.Output;
using Microsoft.ML.Models.BERT.Tokenizers;
using static Microsoft.ML.Models.BERT.BartSummarizeModelConfiguration;

namespace Microsoft.ML.Models.BERT
{
    public class BertQuestionModel : IDisposable
    {
        private readonly BertQuestionModelConfiguration _bertModelConfiguration;
        private List<string> _vocabulary;
        private WordPieceTokenizer _wordPieceTokenizer;
        private PredictionEngine<BertFeature, BertPredictionResult> _predictionEngine;

        public BertQuestionModel(BertQuestionModelConfiguration bertModelConfiguration)
        {
            _bertModelConfiguration = bertModelConfiguration;
        }

        public void Initialize()
        {
            _vocabulary = ReadVocabularyFile(_bertModelConfiguration.VocabularyFile);
            _wordPieceTokenizer = new WordPieceTokenizer(_vocabulary);

            var onnxModelConfigurator = new OnnxModelConfigurator<BertFeature>(_bertModelConfiguration);
            _predictionEngine = onnxModelConfigurator.GetMlNetPredictionEngine<BertPredictionResult>();
        }

        public (List<string> tokens, float probability) Predict(string context, string question)
        {
            var tokens = _wordPieceTokenizer.Tokenize(question, context);
            var encodedFeature = Encode(tokens);

            var result = _predictionEngine.Predict(encodedFeature);
            var contextStart = tokens.FindIndex(o => o.Token == WordPieceTokenizer.DefaultTokens.Separation);

            var (startIndex, endIndex, probability) = GetBestPredictionFromResult(result, contextStart);

            var predictedTokens = encodedFeature.InputIds
                .Skip(startIndex)
                .Take(endIndex + 1 - startIndex)
                .Select(o => _vocabulary[(int)o])
                .ToList();

            var stichedTokens = StitchSentenceBackTogether(predictedTokens);

            return (stichedTokens, probability);
        }

        private List<string> StitchSentenceBackTogether(List<string> tokens)
        {
            var currentToken = string.Empty;

            tokens.Reverse();

            var tokensStitched = new List<string>();

            foreach (var token in tokens)
            {
                if (!token.StartsWith("##"))
                {
                    currentToken = token + currentToken;
                    tokensStitched.Add(currentToken);
                    currentToken = string.Empty;
                }
                else
                {
                    currentToken = token.Replace("##", "") + currentToken;
                }
            }

            tokensStitched.Reverse();

            return tokensStitched;
        }

        private (int StartIndex, int EndIndex, float Probability) GetBestPredictionFromResult(BertPredictionResult result, int minIndex)
        {
            var bestN = _bertModelConfiguration.BestResultSize;

            var bestStartLogits = result.StartLogits
                .Select((logit, index) => (Logit: logit, Index: index))
                .OrderByDescending(o => o.Logit)
                .Take(bestN);

            var bestEndLogits = result.EndLogits
                .Select((logit, index) => (Logit: logit, Index: index))
                .OrderByDescending(o => o.Logit)
                .Take(bestN);

            var bestResultsWithScore = bestStartLogits
                .SelectMany(startLogit =>
                    bestEndLogits
                    .Select(endLogit =>
                        (
                            StartLogit: startLogit.Index,
                            EndLogit: endLogit.Index,
                            Score: startLogit.Logit + endLogit.Logit
                        )
                     )
                )
                .Where(entry => !(entry.EndLogit < entry.StartLogit || entry.EndLogit - entry.StartLogit > _bertModelConfiguration.MaxAnswerLength || entry.StartLogit == 0 && entry.EndLogit == 0 || entry.StartLogit < minIndex))
                .Take(bestN);

            var (item, probability) = bestResultsWithScore
                .Softmax(o => o.Score)
                .OrderByDescending(o => o.Probability)
                .FirstOrDefault();

            return (StartIndex: item.StartLogit, EndIndex: item.EndLogit, probability);
        }

        private BertFeature Encode(List<(string Token, int Index)> tokens)
        {
            var padding = Enumerable
                .Repeat(0L, _bertModelConfiguration.MaxSequenceLength - tokens.Count)
                .ToList();

            var tokenIndexes = tokens
                .Select(token => (long)token.Index)
                .Concat(padding)
                .ToArray();

            var segmentIndexes = GetSegmentIndexes(tokens)
                .Concat(padding)
                .ToArray();

            var inputMask =
                tokens.Select(o => 1L)
                .Concat(padding)
                .ToArray();

            return new BertFeature()
            {
                InputIds = tokenIndexes,
                SegmentIds = segmentIndexes,
                InputMask = inputMask,
                UniqueIds = new long[] { 0 }
            };
        }

        private IEnumerable<long> GetSegmentIndexes(List<(string token, int index)> tokens)
        {
            var segmentIndex = 0;
            var segmentIndexes = new List<long>();

            foreach (var (token, index) in tokens)
            {
                segmentIndexes.Add(segmentIndex);

                if (token == WordPieceTokenizer.DefaultTokens.Separation)
                {
                    segmentIndex++;
                }
            }

            return segmentIndexes;
        }

        private static List<string> ReadVocabularyFile(string filename)
        {
            var vocabulary = new List<string>();

            using (var reader = new StreamReader(filename))
            {
                string line;

                while ((line = reader.ReadLine()) != null)
                {
                    if (!string.IsNullOrWhiteSpace(line))
                    {
                        vocabulary.Add(line);
                    }
                }
            }

            return vocabulary;
        }

        public void Dispose()
        {
            _predictionEngine.Dispose();
        }
    }

    public class BartSummaryModel : IDisposable
    {
        private readonly BartSummarizeModelConfiguration _bartModelConfiguration;
        private List<string> _vocabulary;
        private WordPieceTokenizer _wordPieceTokenizer;
        private PredictionEngine<BartFeature, BartPredictionResult> _predictionEngine;

        public BartSummaryModel(BartSummarizeModelConfiguration bartModelConfiguration)
        {
            _bartModelConfiguration = bartModelConfiguration;
        }

        public void Initialize()
        {
            _vocabulary = ReadVocabularyFile(_bartModelConfiguration.VocabularyFile);
            _wordPieceTokenizer = new WordPieceTokenizer(_vocabulary);

            var onnxModelConfigurator = new OnnxModelConfigurator<BartFeature>(_bartModelConfiguration, new Dictionary<string, int[]>()
                        {
                           { "input_ids", new[] { 1, 1024 } },
                           { "attention_mask", new[] { 1, 1024} },
                           { "decoder_input_ids", new[] { 1, 1024} },
                           { "decoder_attention_mask", new[] { 1, 1024} },
                           { "last_hidden_state", new[] { 1, 1024, 1024} },
                           { "onnx::MatMul_2374", new[] { 1, 1024, 1024} }
                        });

            _predictionEngine = onnxModelConfigurator.GetMlNetPredictionEngine<BartPredictionResult>();
        }

        public (List<string> tokens, float probability) Predict(string context, string question)
        {
            var tokens = _wordPieceTokenizer.Tokenize(question, context);
            var encodedFeature = EncodeBart(tokens);

            var result = _predictionEngine.Predict(encodedFeature);
            var contextStart = tokens.FindIndex(o => o.Token == WordPieceTokenizer.DefaultTokens.Separation);

            var (startIndex, endIndex, probability) = GetBestPredictionFromResult(result, contextStart);

            var predictedTokens = encodedFeature.InputIds
                .Skip(startIndex)
                .Take(endIndex + 1 - startIndex)
                .Select(o => _vocabulary[(int)o])
                .ToList();

            var stichedTokens = StitchSentenceBackTogether(predictedTokens);

            return (stichedTokens, probability);
        }

        public (List<string> tokens, float probability) PredictBart(string context)
        {
            var tokens = _wordPieceTokenizer.Tokenize(context);
            var encodedFeature = EncodeBart(tokens);

            var result = _predictionEngine.Predict(encodedFeature);
            var contextStart = tokens.FindIndex(o => o.Token == WordPieceTokenizer.DefaultTokens.Separation);

            var (startIndex, endIndex, probability) = GetBestPredictionFromResult(result, contextStart);

            var predictedTokens = encodedFeature.InputIds
                .Skip(startIndex)
                .Take(endIndex + 1 - startIndex)
                .Select(o => _vocabulary[(int)o])
                .ToList();

            var stichedTokens = StitchSentenceBackTogether(predictedTokens);

            return (stichedTokens, probability);
        }

        private List<string> StitchSentenceBackTogether(List<string> tokens)
        {
            var currentToken = string.Empty;

            tokens.Reverse();

            var tokensStitched = new List<string>();

            foreach (var token in tokens)
            {
                if (!token.StartsWith("##"))
                {
                    currentToken = token + currentToken;
                    tokensStitched.Add(currentToken);
                    currentToken = string.Empty;
                }
                else
                {
                    currentToken = token.Replace("##", "") + currentToken;
                }
            }

            tokensStitched.Reverse();

            return tokensStitched;
        }

        private (int StartIndex, int EndIndex, float Probability) GetBestPredictionFromResult(BPredictionResult result, int minIndex)
        {
            var bestN = 25;

            var bestStartLogits = result.StartLogits
                .Select((logit, index) => (Logit: logit, Index: index))
                .OrderByDescending(o => o.Logit)
                .Take(bestN);

            var bestEndLogits = result.EndLogits
                .Select((logit, index) => (Logit: logit, Index: index))
                .OrderByDescending(o => o.Logit)
                .Take(bestN);

            var bestResultsWithScore = bestStartLogits
                .SelectMany(startLogit =>
                    bestEndLogits
                    .Select(endLogit =>
                        (
                            StartLogit: startLogit.Index,
                            EndLogit: endLogit.Index,
                            Score: startLogit.Logit + endLogit.Logit
                        )
                     )
                )
                .Where(entry => !(entry.EndLogit < entry.StartLogit || entry.EndLogit - entry.StartLogit > _bartModelConfiguration.MaxAnswerLength || entry.StartLogit == 0 && entry.EndLogit == 0 || entry.StartLogit < minIndex))
                .Take(bestN);

            var (item, probability) = bestResultsWithScore
                .Softmax(o => o.Score)
                .OrderByDescending(o => o.Probability)
                .FirstOrDefault();

            return (StartIndex: item.StartLogit, EndIndex: item.EndLogit, probability);
        }

        private BertFeature Encode(List<(string Token, int Index)> tokens)
        {
            var padding = Enumerable
                .Repeat(0L, _bartModelConfiguration.MaxSequenceLength - tokens.Count)
                .ToList();

            var tokenIndexes = tokens
                .Select(token => (long)token.Index)
                .Concat(padding)
                .ToArray();

            var segmentIndexes = GetSegmentIndexes(tokens)
                .Concat(padding)
                .ToArray();

            var inputMask =
                tokens.Select(o => 1L)
                .Concat(padding)
                .ToArray();

            return new BertFeature()
            {
                InputIds = tokenIndexes,
                SegmentIds = segmentIndexes,
                InputMask = inputMask,
                UniqueIds = new long[] { 0 }
            };
        }

        private BartFeature EncodeBart(List<(string Token, int Index)> tokens)
        {
            var padding = Enumerable
                .Repeat(0L, _bartModelConfiguration.MaxSequenceLength - tokens.Count)
                .ToList();

            var tokenIndexes = tokens
                .Select(token => (long)token.Index)
                .Concat(padding)
                .ToArray();

            var segmentIndexes = GetSegmentIndexes(tokens)
                .Concat(padding)
                .ToArray();

            var inputMask =
                tokens.Select(o => 1L)
                .Concat(padding)
                .ToArray();

            return new BartFeature()
            {
                InputIds = tokenIndexes,
                SegmentIds = segmentIndexes,
                InputMask = inputMask,
                UniqueIds = new long[] { 0 }
            };
        }

        private IEnumerable<long> GetSegmentIndexes(List<(string token, int index)> tokens)
        {
            var segmentIndex = 0;
            var segmentIndexes = new List<long>();

            foreach (var (token, index) in tokens)
            {
                segmentIndexes.Add(segmentIndex);

                if (token == WordPieceTokenizer.DefaultTokens.Separation)
                {
                    segmentIndex++;
                }
            }

            return segmentIndexes;
        }

        private static List<string> ReadVocabularyFile(string filename)
        {
            var vocabulary = new List<string>();

            using (var reader = new StreamReader(filename))
            {
                string line;

                while ((line = reader.ReadLine()) != null)
                {
                    if (!string.IsNullOrWhiteSpace(line))
                    {
                        vocabulary.Add(line);
                    }
                }
            }

            return vocabulary;
        }

        public void Dispose()
        {
            _predictionEngine.Dispose();
        }
    }

    /*public class BartSummaryModel
    {
        string bartModelPath, vocabPath;
        MLContext _mlContext;

        public void Initialize(string path, string vocab)
        {
            _mlContext = new MLContext();

            bartModelPath = path;
            vocabPath = vocab;
        }

        public void Predict()
        {
            var pipeline = _mlContext.Transforms
                            .ApplyOnnxModel(modelFile: bartModelPath,
                                            shapeDictionary: new Dictionary<string, int[]>
                                            {
                                                { "input_ids", new [] { 1, 32 } },
                                                { "attention_mask", new [] { 1, 32 } },
                                                { "decoder_input_ids", new [] { 1, 32 } },
                                                { "decoder_attention_mask", new [] { 1, 32 } },
                                                { "last_hidden_state", new [] { 1, 32, 1024 } },
                                                { "onnx::MatMul_2374", new [] { 1, 32 , 1024 } },
                                            },
                                            inputColumnNames: new[] {"input_ids",
                                                                     "attention_mask",
                                                                     "decoder_input_ids",
                                                                     "decoder_attention_mask"},
                                            outputColumnNames: new[] { "last_hidden_state",
                                                                       "onnx::MatMul_2374"},
                                            gpuDeviceId: false ? 0 : (int?)null,
                                            fallbackToCpu: true);

            var model = pipeline.Fit(_mlContext.Data.LoadFromEnumerable(new List<ModelInput>()));
        }
    }*/
}
