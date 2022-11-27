using Microsoft.ML.Data;
using Microsoft.ML.Models.BERT.Onnx;

namespace Microsoft.ML.Models.BERT
{
    public class BertQuestionModelConfiguration : IOnnxModel
    {
        public int MaxSequenceLength { get; set; } = 256;

        public int MaxAnswerLength { get; set; } = 30;

        public int BestResultSize { get; set; } = 20;

        public string VocabularyFile { get; set; }

        public string ModelPath { get; set; }

        public string[] ModelInput => new[] { "unique_ids_raw_output___9:0", "segment_ids:0", "input_mask:0", "input_ids:0" };

        public string[] ModelOutput => new[] { "unstack:1", "unstack:0", "unique_ids:0" };
    }

    public class BartSummarizeModelConfiguration : IOnnxModel
    {
        public int MaxSequenceLength { get; set; } = 1024;

        public int MaxAnswerLength { get; set; } = 1024;

        public string VocabularyFile { get; set; }

        public string ModelPath { get; set; }

        public string[] ModelInput => new[] { "input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask" };

        public string[] ModelOutput => new[] { "last_hidden_state", "onnx::MatMul_2374" };
    }
}