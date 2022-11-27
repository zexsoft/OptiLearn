using Microsoft.ML.Data;
using System.Collections.Generic;

namespace Microsoft.ML.Models.BERT.Input
{
    internal class BFeature
    {
        public virtual long[] UniqueIds { get; set; }

        public virtual long[] SegmentIds { get; set; }

        public virtual long[] InputMask { get; set; }

        public virtual long[] InputIds { get; set; }
    }

    internal class BertFeature : BFeature
    {
        [VectorType(1)]
        [ColumnName("unique_ids_raw_output___9:0")]
        public override long[] UniqueIds { get; set; }

        [VectorType(1, 256)]
        [ColumnName("segment_ids:0")]
        public override long[] SegmentIds { get; set; }

        [VectorType(1, 256)]
        [ColumnName("input_mask:0")]
        public override long[] InputMask { get; set; }

        [VectorType(1, 256)]
        [ColumnName("input_ids:0")]
        public override long[] InputIds { get; set; }
    }

    internal class BartFeature : BFeature
    {
        [VectorType(1, 1024)]
        [ColumnName("input_ids")]
        public override long[] InputIds { get; set; }

        [VectorType(1, 1024)]
        [ColumnName("attention_mask")]
        public override long[] InputMask { get; set; }

        [VectorType(1, 1024)]
        [ColumnName("decoder_input_ids")]
        public override long[] UniqueIds { get; set; }

        [VectorType(1, 1024)]
        [ColumnName("decoder_attention_mask")]
        public override long[] SegmentIds { get; set; }
    }
}
