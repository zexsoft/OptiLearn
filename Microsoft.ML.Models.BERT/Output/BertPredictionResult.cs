using Microsoft.ML.Data;

namespace Microsoft.ML.Models.BERT.Output
{
    internal class BPredictionResult
    {
        public virtual float[] EndLogits { get; set; }

        public virtual float[] StartLogits { get; set; }

        public virtual long[] UniqueIds { get; set; }
    }

    internal class BertPredictionResult : BPredictionResult
    {
        [VectorType(1, 256)]
        [ColumnName("unstack:1")]
        public override float[] EndLogits { get; set; }

        [VectorType(1, 256)]
        [ColumnName("unstack:0")]
        public override float[] StartLogits { get; set; }

        [VectorType(1)]
        [ColumnName("unique_ids:0")]
        public override long[] UniqueIds { get; set; }
    }

    internal class BartPredictionResult : BPredictionResult
    {
        [VectorType(1, 1024)]
        [ColumnName("last_hidden_state")]
        public override float[] EndLogits { get; set; }

        [VectorType(1, 1024)]
        [ColumnName("onnx::MatMul_2374")]
        public override float[] StartLogits { get; set; }
    }
}
