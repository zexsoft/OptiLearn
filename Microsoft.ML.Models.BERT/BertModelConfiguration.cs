﻿using Microsoft.ML.Models.BERT.Onnx;

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

    public class BertSummarizeModelConfiguration : IOnnxModel
    {
        public int MaxSequenceLength { get; set; } = 1024;

        public int MaxAnswerLength { get; set; } = 1024;

        public int BestResultSize { get; set; } = 20;

        public string VocabularyFile { get; set; }

        public string ModelPath { get; set; }

        public string[] ModelInput => new[] { "input_ids:0", "attention_mask:0", "decoder_input_ids:0", "decoder_attention_mask:0" };

        public string[] ModelOutput => new[] { "last_hidden_state:1", "onnx::MatMul_2374:0" };
    }
}